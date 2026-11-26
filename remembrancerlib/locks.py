"""Cross-process serialization for workflows that mutate Remembrancer's git worktree."""
from __future__ import annotations

import fcntl
import functools
import inspect
import os
import stat
import threading
from contextlib import contextmanager


class PipelineBusyError(RuntimeError):
    """Raised when an interactive writer would race nightly/catch-up."""


_LOCAL = threading.local()


def open_repository_state_directory(
    root: str, *, create: bool,
) -> tuple[int, bool]:
    """Open ``root/state`` without following either path component symlink.

    Callers own the returned descriptor.  Creating the directory also fsyncs
    the repository directory so a crash cannot lose the new directory entry.
    """
    root = os.path.abspath(root)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_descriptor = os.open(root, directory_flags)
    created = False
    try:
        if create:
            try:
                os.mkdir("state", 0o700, dir_fd=root_descriptor)
                created = True
                os.fsync(root_descriptor)
            except FileExistsError:
                pass
        try:
            state_descriptor = os.open(
                "state", directory_flags, dir_fd=root_descriptor,
            )
        except FileNotFoundError:
            if create:
                raise
            return -1, False
        if not stat.S_ISDIR(os.fstat(state_descriptor).st_mode):
            os.close(state_descriptor)
            raise NotADirectoryError("repository state is not a directory")
        return state_descriptor, created
    finally:
        os.close(root_descriptor)


def pipeline_mutation_lock_held(root: str) -> bool:
    """Whether this thread is inside the repository's mutation transaction.

    The nightly parent owns the actual ``flock`` and marks its children with
    ``REMEMBRANCER_PIPELINE_LOCK_HELD``; interactive workflows are tracked in
    thread-local state by :func:`pipeline_mutation_lock`.
    """
    if os.environ.get("REMEMBRANCER_PIPELINE_LOCK_HELD") == "1":
        return True
    lock_path = os.path.join(os.path.abspath(root), "state", "pipeline.lock")
    held = getattr(_LOCAL, "held", {})
    return int(held.get(lock_path, 0)) > 0


@contextmanager
def pipeline_mutation_lock(root: str, *, blocking: bool = False):
    """Hold ``state/pipeline.lock`` for one complete external write workflow.

    The lock is re-entrant within one thread so a UI workflow can call a lower
    level producer that uses the same guard. Nightly owns the lock in its parent
    shell and exports ``REMEMBRANCER_PIPELINE_LOCK_HELD=1`` to child processes; those
    children must not acquire a second flock description.
    """
    root = os.path.abspath(root)
    if os.environ.get("REMEMBRANCER_PIPELINE_LOCK_HELD") == "1":
        yield
        return

    lock_path = os.path.join(root, "state", "pipeline.lock")
    held = getattr(_LOCAL, "held", {})
    depth = int(held.get(lock_path, 0))
    if depth:
        held[lock_path] = depth + 1
        _LOCAL.held = held
        try:
            yield
        finally:
            held[lock_path] -= 1
        return

    state_descriptor, _created = open_repository_state_directory(
        root, create=True,
    )
    file_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_descriptor = os.open(
            "pipeline.lock", file_flags, 0o600, dir_fd=state_descriptor,
        )
    except BaseException:
        os.close(state_descriptor)
        raise
    try:
        lock = os.fdopen(lock_descriptor, "a+")
    except BaseException:
        os.close(lock_descriptor)
        os.close(state_descriptor)
        raise
    try:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(lock.fileno(), flags)
        except BlockingIOError as exc:
            raise PipelineBusyError(
                "memory pipeline is busy; retry after nightly/catch-up finishes"
            ) from exc
        held[lock_path] = 1
        _LOCAL.held = held
        try:
            yield
        finally:
            held.pop(lock_path, None)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    finally:
        lock.close()
        os.close(state_descriptor)


@contextmanager
def artifact_publish_lock(root: str, *, blocking: bool = True):
    """Serialize artifact publishes against EACH OTHER — never against nightly.

    A separate inode from ``pipeline.lock``: nightly never acquires this, so a
    running nightly/wiki transaction can never make publishing report busy.
    """
    root = os.path.abspath(root)
    state_descriptor, _created = open_repository_state_directory(root, create=True)
    file_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_descriptor = os.open(
            "artifact_publish.lock", file_flags, 0o600, dir_fd=state_descriptor,
        )
    except BaseException:
        os.close(state_descriptor)
        raise
    try:
        lock = os.fdopen(lock_descriptor, "a+")
    except BaseException:
        os.close(lock_descriptor)
        os.close(state_descriptor)
        raise
    try:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(lock.fileno(), flags)
        except BlockingIOError as exc:
            raise PipelineBusyError("artifact publish is busy") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    finally:
        lock.close()
        os.close(state_descriptor)


@contextmanager
def remember_spool_lock(root: str, *, blocking: bool = True):
    """Serialize agent-write spooling against ITSELF — never against nightly.

    Same rationale (and same separate-inode trick) as ``artifact_publish_lock``:
    the nightly/catch-up transaction never acquires this, so a running pipeline
    can never make ``remembrancer_remember`` report busy. A catch-up run can last as
    long as its timer interval, so the pipeline lock can be held almost
    continuously; sharing it would make every agent write fail.
    """
    root = os.path.abspath(root)
    state_descriptor, _created = open_repository_state_directory(root, create=True)
    file_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_descriptor = os.open(
            "remember_spool.lock", file_flags, 0o600, dir_fd=state_descriptor,
        )
    except BaseException:
        os.close(state_descriptor)
        raise
    try:
        lock = os.fdopen(lock_descriptor, "a+")
    except BaseException:
        os.close(lock_descriptor)
        os.close(state_descriptor)
        raise
    try:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(lock.fileno(), flags)
        except BlockingIOError as exc:
            raise PipelineBusyError("remember spool is busy") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    finally:
        lock.close()
        os.close(state_descriptor)


@contextmanager
def approval_intents_lock(root: str, *, blocking: bool = False):
    """Single-writer guard for the signed-approval applier — never nightly's.

    Same separate-inode trick as ``remember_spool_lock``: the pipeline never
    takes ``approval_intents.lock``, so holding it says nothing about
    ``pipeline.lock``. It only keeps two appliers (a second worker or process)
    from draining the intent ledger at once; each still takes the pipeline lock
    for the apply itself. Non-blocking by default: a held lock means another
    applier is already working, and the caller skips this tick.
    """
    root = os.path.abspath(root)
    state_descriptor, _created = open_repository_state_directory(root, create=True)
    file_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_descriptor = os.open(
            "approval_intents.lock", file_flags, 0o600, dir_fd=state_descriptor,
        )
    except BaseException:
        os.close(state_descriptor)
        raise
    try:
        lock = os.fdopen(lock_descriptor, "a+")
    except BaseException:
        os.close(lock_descriptor)
        os.close(state_descriptor)
        raise
    try:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(lock.fileno(), flags)
        except BlockingIOError as exc:
            raise PipelineBusyError("approval intent applier is busy") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    finally:
        lock.close()
        os.close(state_descriptor)


def serialized_pipeline_mutation(*, root_argument: str = "root",
                                 blocking: bool = False):
    """Decorate a public mutation whose repository root is a named argument."""
    def decorate(function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def guarded(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            root = bound.arguments.get(root_argument, ".")
            with pipeline_mutation_lock(str(root), blocking=blocking):
                return function(*args, **kwargs)

        return guarded
    return decorate

"""A client that ends its MCP session normally is not an error in the server log.

mcp 1.30.0 logs that close as ``Error in standalone SSE writer`` with an
``anyio.ClosedResourceError`` traceback (modelcontextprotocol/python-sdk#3833).
Importing mcp/server.py, as running the server does, puts a filter on the
SDK's logger that makes exactly that record DEBUG. These tests log the record
the SDK logs, on the SDK's own logger, and read what reaches a handler that,
like the one FastMCP installs, has no level of its own.
"""
import logging

import anyio
import pytest

import mcp.server  # noqa: F401  (installs the filter)

SDK_LOGGER = "mcp.server.streamable_http"
WRITER_ERROR = "Error in standalone SSE writer"


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.NOTSET)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def sdk_log():
    """The SDK's logger at ``level``, and the records a level-less handler gets."""
    logger = logging.getLogger(SDK_LOGGER)
    handler, before = _Collect(), logger.level

    def at(level):
        logger.setLevel(level)
        return handler.records
    logger.addHandler(handler)
    yield at
    logger.removeHandler(handler)
    logger.setLevel(before)


def _writer_stops(exc, message=WRITER_ERROR):
    """Log what the SDK's SSE writer logs when ``exc`` stops it."""
    try:
        raise exc
    except Exception:
        logging.getLogger(SDK_LOGGER).exception(message)


def _seen(records):
    return [(r.levelname, r.getMessage()) for r in records]


def test_a_normal_session_close_leaves_no_record(sdk_log):
    records = sdk_log(logging.INFO)
    _writer_stops(anyio.ClosedResourceError())
    assert _seen(records) == []


def test_at_debug_the_close_is_still_logged(sdk_log):
    records = sdk_log(logging.DEBUG)
    _writer_stops(anyio.ClosedResourceError())
    assert _seen(records) == [("DEBUG", WRITER_ERROR)]
    assert records[0].exc_info[0] is anyio.ClosedResourceError


@pytest.mark.parametrize("exc, message", [
    (RuntimeError("the writer broke"), WRITER_ERROR),
    (anyio.ClosedResourceError(), "Error in SSE writer"),
    (anyio.ClosedResourceError(), "Error in standalone SSE response"),
])
def test_any_other_writer_error_stays_an_error(sdk_log, exc, message):
    records = sdk_log(logging.INFO)
    _writer_stops(exc, message)
    assert _seen(records) == [("ERROR", message)]

import json

import pytest

from remembrancerlib import Cursors


def test_stale_instances_merge_instead_of_losing_other_cursor(tmp_path):
    path = str(tmp_path / "state" / "cursors.json")
    first = Cursors(path)
    second = Cursors(path)

    first.set("gmail:a", "100")
    second.set("gcal:b", "200")

    assert json.loads((tmp_path / "state" / "cursors.json").read_text()) == {
        "gmail:a": "100", "gcal:b": "200"
    }
    assert first.get("gcal:b") == "200"


def test_corrupt_cursor_store_fails_closed(tmp_path):
    path = tmp_path / "cursors.json"
    path.write_text("{broken")
    with pytest.raises(RuntimeError, match="corrupt"):
        Cursors(str(path))

"""Response-letter skeleton: every comment and lettered sub-point becomes an
entry with its concern verbatim, and filled entries survive a re-run."""
import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LETTER = """Reviewers' Comments to Author:

Reviewer: 1

Comments:
Summary paragraph.

1. First concern, line one
continues here.

2. Second concern.

Additional Questions:
1) Does the paper contribute to the body of knowledge?: Partially, because reasons.

2) Is the paper technically sound?: No

Reviewer: 2

Comments:
MAJOR COMMENTS

1. Heading of item one.
  (a) Sub-point a.
  (b) Sub-point b.

MINOR COMMENTS

- A minor point.

Additional Questions:
1) Does the paper contribute to the body of knowledge?: Yes
"""


def test_extract_and_merge(tmp_path):
    spec = importlib.util.spec_from_file_location("erc", ROOT / "scripts" / "extract_review_comments.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    (tmp_path / "l.txt").write_text(LETTER, encoding="utf-8")
    out = tmp_path / "r.yaml"
    assert m.main(["--letter", str(tmp_path / "l.txt"), "--out", str(out)]) == 0
    entries = {e["id"]: e for e in yaml.safe_load(out.read_text(encoding="utf-8"))["entries"]}
    assert entries["R1.1"]["concern"] == "1. First concern, line one\ncontinues here."
    assert entries["R1.q1"]["concern"] == "Partially, because reasons."
    assert "R1.q2" not in entries                                    # bare "No" carries no concern
    assert entries["R2.1"]["kind"] == "heading"
    assert entries["R2.1a"]["concern"] == "(a) Sub-point a." and entries["R2.1b"]["kind"] == "comment"
    assert entries["R2.m1"]["concern"] == "A minor point."
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    doc["entries"][1]["response"] = "filled"
    out.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    m.main(["--letter", str(tmp_path / "l.txt"), "--out", str(out)])
    again = yaml.safe_load(out.read_text(encoding="utf-8"))["entries"]
    assert again[1]["response"] == "filled" and len(again) == len(entries)

"""This repository is public: nothing tracked names a private machine. A benchmark note says what hardware
CLASS a measurement ran on (a CPU host, a GH200, an L40S -- all fine to name) and never which machine.

The forbidden tokens themselves are never written here, in a comment, in a commit message or in test output:
publishing a byte-exact list of "the names we must not say" defeats the point. Instead this holds salted
SHA-256 digests of the (lowercased) tokens; a file is flagged by tokenising it on ``[A-Za-z0-9-]+`` and hashing
each token the same way. A failure reports which FILE matched, never which word."""
import hashlib
import os
import re
import subprocess

SALT = "dmrai-lab/disco-space:no-private-identifiers:v1"

#: salted sha256(SALT + token.lower()) of each forbidden token -- never the token itself
FORBIDDEN_DIGESTS = frozenset((
    "332482b751888461b8dd6f0f6b988fa3524fb79cdf9cbdacb4177c9ed7ffa77d",
    "8acbc150f613da285bda4dfe2c932e5d64d6d5b6f130fdbe8c6d0d6350d642bf",
    "90964afc5e136e5912547b5bb0a9d3c6d235aa4f7e3defbbda5cd8f8fe20a1b2",
))

TOKEN_RE = re.compile(r"[A-Za-z0-9-]+")


def _digest(token):
    return hashlib.sha256((SALT + token.lower()).encode()).hexdigest()


def tracked_files():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(["git", "-C", root, "ls-files"], capture_output=True, text=True, check=True)
    return [os.path.join(root, p) for p in out.stdout.splitlines() if p]


def test_no_tracked_file_names_a_private_machine():
    hits = []
    for path in tracked_files():
        if os.path.abspath(path) == os.path.abspath(__file__):
            continue                                  # this file's own digest table
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, IsADirectoryError):
            continue                                  # a binary asset (data/*.nii.gz, images, ...)
        if any(_digest(tok) in FORBIDDEN_DIGESTS for tok in TOKEN_RE.findall(text)):
            hits.append(path)
    # the message names the FILE only -- never the matched word, which stays undisclosed even on failure
    assert not hits, f"{len(hits)} tracked file(s) name a forbidden private machine: {hits}"

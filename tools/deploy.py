"""Deploy this checkout to the Space: the files the image needs (no tests, tools or workflows), one commit on the
Space repo naming the source commit; Hugging Face rebuilds the image on that push.

    HF_TOKEN=... python tools/deploy.py [--space rfick/disco] [--sha <git sha>]
"""
import argparse
import os
import subprocess

IGNORE = [".git/*", "__pycache__/*", "*.pyc", ".github/*", "tests/*", "tools/*", ".pytest_cache/*", ".gitignore", "gate.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=os.environ.get("DISCO_SPACE", "rfick/disco"))
    ap.add_argument("--sha", default=None)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sha = a.sha or subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    info = api.upload_folder(folder_path=root, repo_id=a.space, repo_type="space", ignore_patterns=IGNORE,
                             commit_message=f"disco-space {sha[:12]}")
    print(info.commit_url)
    print("stage", api.get_space_runtime(a.space).stage)


if __name__ == "__main__":
    main()

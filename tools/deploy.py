"""Deploy this checkout to the Space: the files the image needs (workflows left out), one commit on the Space repo
naming the source commit; Hugging Face rebuilds the image on that push. With ``--base-tag`` the Space's Dockerfile
is ``Dockerfile.space`` on the prebuilt ``ghcr.io/dmrai-lab/disco-space:<tag>`` (the image workflow's output), so
the build copies files instead of installing the stack.

    HF_TOKEN=... python tools/deploy.py [--space rfick/disco] [--sha <git sha>] [--base-tag <sha|latest>] [--zero]

Both Spaces read the layout from the Hub at the config's revision when their container starts (disco-space#4: the
bucket is retired), so the deploy also removes a ``DISCO_MOMENTS`` variable and any mounted volume it finds.
"""
import argparse
import os
import shutil
import subprocess
import tempfile

IGNORE = [".git/*", "__pycache__/*", "*.pyc", ".github/*", ".pytest_cache/*", ".gitignore", "gate.json"]   # tests and tools go too: the gate job runs on the Space's own files


def retire_bucket(api, space):
    """No ``DISCO_MOMENTS`` variable and no volume on ``space``: the app then downloads the layout from the Hub."""
    try:
        api.delete_space_variable(space, "DISCO_MOMENTS")
        print("removed the DISCO_MOMENTS variable")
    except Exception as e:                                  # absent already, or the API says so
        print("DISCO_MOMENTS variable:", repr(e)[:120])
    try:
        api.delete_space_volumes(space)
        print("no volumes mounted")
    except Exception as e:
        print("volumes:", repr(e)[:160])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=os.environ.get("DISCO_SPACE", "rfick/disco"))
    ap.add_argument("--sha", default=None)
    ap.add_argument("--base-tag", default=None, help="deploy on ghcr.io/dmrai-lab/disco-space:<tag> through Dockerfile.space")
    ap.add_argument("--zero", action="store_true", help="the ZeroGPU (Gradio SDK) deployment: README-zero.md, requirements-zero.txt, app.py")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sha = a.sha or subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    folder = root; staged = None
    if a.zero:
        staged = tempfile.mkdtemp(prefix="disco-zero-")
        for name in ("space", "data", "tests", "tools"):
            shutil.copytree(os.path.join(root, name), os.path.join(staged, name), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("app.py", "pyproject.toml"):
            shutil.copy(os.path.join(root, name), staged)
        shutil.copy(os.path.join(root, "README-zero.md"), os.path.join(staged, "README.md"))
        shutil.copy(os.path.join(root, "requirements-zero.txt"), os.path.join(staged, "requirements.txt"))
        folder = staged
    elif a.base_tag:
        staged = tempfile.mkdtemp(prefix="disco-space-")
        for name in ("space", "data", "tests", "tools"):
            shutil.copytree(os.path.join(root, name), os.path.join(staged, name), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("README.md", "requirements.txt", "entrypoint.sh", "pyproject.toml"):
            shutil.copy(os.path.join(root, name), staged)
        with open(os.path.join(root, "Dockerfile.space")) as f, open(os.path.join(staged, "Dockerfile"), "w") as g:
            g.write(f.read().replace("BASE_TAG", a.base_tag))
        folder = staged
    info = api.upload_folder(folder_path=folder, repo_id=a.space, repo_type="space", ignore_patterns=IGNORE,
                             delete_patterns=["Dockerfile", "Dockerfile.base", "Dockerfile.space", "entrypoint.sh", "README-zero.md", "requirements-zero.txt"] if a.zero
                             else (["Dockerfile.base", "Dockerfile.space"] if a.base_tag else None),
                             commit_message=f"disco-space {sha[:12]}" + (f" on base {a.base_tag[:12]}" if a.base_tag else ""))
    if staged:
        shutil.rmtree(staged)
    retire_bucket(api, a.space)
    print(info.commit_url)
    print("stage", api.get_space_runtime(a.space).stage)


if __name__ == "__main__":
    main()

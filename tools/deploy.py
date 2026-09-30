"""Deploy this checkout to the Space: the files the image needs (workflows left out), one commit on the Space repo
naming the source commit; Hugging Face rebuilds the image on that push. With ``--base-tag`` the Space's Dockerfile
is ``Dockerfile.space`` on the prebuilt ``ghcr.io/dmrai-lab/disco-space:<tag>`` (the image workflow's output), so
the build copies files instead of installing the stack.

    HF_TOKEN=... python tools/deploy.py [--space rfick/disco] [--sha <git sha>] [--base-tag <sha|latest>] [--zero] [--config brain.toml]

``--config`` names the configuration the Space serves (a file in ``space/``; ``config.toml``, the DiSCo Space, by
default): the deploy sets the Space variable ``DISCO_CONFIG`` to it (and removes it for the default), and a ZeroGPU
deployment takes ``README-zero.md`` for the default and ``README-<stem>-zero.md`` for another (``brain.toml`` ->
``README-brain-zero.md``). ``rfick/brain-zero`` is this repository deployed with ``--zero --config brain.toml``.

The Spaces read their data from the Hub at the config's revision when their container starts (disco-space#4: the
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
        print("volumes removed")
    except Exception as e:                                  # "has no attached volumes" is the state we want
        print("no volumes mounted" if "no attached" in str(e) else f"volumes: {e!r}"[:160])


DEFAULT_CONFIG = "config.toml"


def readme_for(config):
    """The ZeroGPU README of a configuration: ``README-zero.md`` for the default, ``README-<stem>-zero.md`` else."""
    return "README-zero.md" if config == DEFAULT_CONFIG else f"README-{os.path.splitext(config)[0]}-zero.md"


def set_config(api, space, config):
    """``DISCO_CONFIG`` on the Space: the configuration's name, absent for the default."""
    if config == DEFAULT_CONFIG:
        try:
            api.delete_space_variable(space, "DISCO_CONFIG")
            print("removed the DISCO_CONFIG variable (the default configuration)")
        except Exception as e:                              # absent already
            print("DISCO_CONFIG variable:", repr(e)[:120])
    else:
        api.add_space_variable(space, "DISCO_CONFIG", config)
        print(f"DISCO_CONFIG = {config}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=os.environ.get("DISCO_SPACE", "rfick/disco"))
    ap.add_argument("--sha", default=None)
    ap.add_argument("--base-tag", default=None, help="deploy on ghcr.io/dmrai-lab/disco-space:<tag> through Dockerfile.space")
    ap.add_argument("--zero", action="store_true", help="the ZeroGPU (Gradio SDK) deployment: README-zero.md, requirements-zero.txt, app.py")
    ap.add_argument("--config", default=os.environ.get("DISCO_CONFIG", DEFAULT_CONFIG), help="the configuration in space/ the Space serves (DISCO_CONFIG)")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(os.path.join(root, "space", a.config)):
        raise SystemExit(f"no configuration space/{a.config}")
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
        shutil.copy(os.path.join(root, readme_for(a.config)), os.path.join(staged, "README.md"))
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
                             delete_patterns=["Dockerfile", "Dockerfile.base", "Dockerfile.space", "entrypoint.sh", "README-*zero.md", "requirements-zero.txt"] if a.zero
                             else (["Dockerfile.base", "Dockerfile.space"] if a.base_tag else None),
                             commit_message=f"disco-space {sha[:12]}" + (f" on base {a.base_tag[:12]}" if a.base_tag else "") + (f", {a.config}" if a.config != DEFAULT_CONFIG else ""))
    if staged:
        shutil.rmtree(staged)
    set_config(api, a.space, a.config)
    retire_bucket(api, a.space)
    print(info.commit_url)
    print("stage", api.get_space_runtime(a.space).stage)


if __name__ == "__main__":
    main()

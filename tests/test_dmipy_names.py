"""Every dmipy name this repository imports resolves in the installed release. A static pass over the imports of the
package, the tools and the tests -- nothing of them is run -- because most sit inside functions that only a deployment
calls: a name that moved between releases fails here, in the gate, not on the Space."""
import ast
import importlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sources():
    for top in ("space", "tools", "tests", "app.py"):
        path = os.path.join(ROOT, top)
        if os.path.isfile(path):
            yield path
            continue
        for base, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            yield from (os.path.join(base, f) for f in sorted(files) if f.endswith(".py"))


def _imports():
    """``(file, line, module, name)`` of every ``import dmipy_x.y`` (``name`` None) and ``from dmipy_x.y import name``."""
    for path in _sources():
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                yield from ((path, node.lineno, a.name, None) for a in node.names if a.name.startswith("dmipy"))
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("dmipy") and not node.level:
                yield from ((path, node.lineno, node.module, a.name) for a in node.names)


def test_every_dmipy_name_the_repository_imports_resolves():
    imports = list(_imports())
    assert imports                                             # the pass reads the package, the tools and the tests
    unresolved = []
    for path, line, module, name in imports:
        try:
            mod = importlib.import_module(module)
            if name is not None and not hasattr(mod, name):
                importlib.import_module(f"{module}.{name}")    # a submodule: `from dmipy_replay.replay import so3`
        except Exception as e:                                 # an ImportError names the moved name; anything else is a broken module
            unresolved.append(f"{os.path.relpath(path, ROOT)}:{line}: {module}{'' if name is None else ' import ' + name}: {e}")
    assert not unresolved, "\n".join(unresolved)

"""The brain Space's requirements are the default's with its own dmipy pins: one family of requirements files."""
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _deploy():
    spec = importlib.util.spec_from_file_location("deploy_tool", os.path.join(ROOT, "tools", "deploy.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pins(text):
    d = _deploy()
    return {d._package_of(l): l for l in text.splitlines() if d._package_of(l)}


def test_the_default_is_requirements_zero_as_it_stands():
    d = _deploy()
    with open(os.path.join(ROOT, "requirements-zero.txt")) as f:
        assert d.compose_requirements(ROOT, d.DEFAULT_CONFIG) == f.read()


def test_the_brain_space_replaces_only_its_dmipy_pins():
    d = _deploy()
    default = _pins(d.compose_requirements(ROOT, d.DEFAULT_CONFIG))
    brain = _pins(d.compose_requirements(ROOT, "brain.toml"))
    assert set(brain) == set(default)
    changed = sorted(k for k in brain if brain[k] != default[k])
    assert changed == ["dmipy-sim", "dmipy-tract"]
    assert "1b79d2e06a958578bc523c84bacf54bae86fbb95" in brain["dmipy-sim"]
    assert "7da22c3a9473c50a08cc7c6246ec16b68a64d877" in brain["dmipy-tract"]

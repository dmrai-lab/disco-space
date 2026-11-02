"""The ZeroGPU requirements of a configuration are the default's, with the pins of its own file when it has one: one
family of requirements files."""
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _deploy():
    spec = importlib.util.spec_from_file_location("deploy_tool", os.path.join(ROOT, "tools", "deploy.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_default_is_requirements_zero_as_it_stands():
    d = _deploy()
    with open(os.path.join(ROOT, "requirements-zero.txt")) as f:
        assert d.compose_requirements(ROOT, d.DEFAULT_CONFIG) == f.read()


def test_the_brain_space_takes_the_default_pins():
    """One code runs both pages, so one release of the dmipy packages serves both: the brain Space has no pins of its own."""
    d = _deploy()
    assert not os.path.exists(os.path.join(ROOT, d.requirements_for("brain.toml")))
    assert d.compose_requirements(ROOT, "brain.toml") == d.compose_requirements(ROOT, d.DEFAULT_CONFIG)


def test_a_configuration_with_pins_of_its_own_replaces_only_those_lines(tmp_path):
    d = _deploy()
    (tmp_path / "requirements-zero.txt").write_text("numpy>=2\n# the stack\njax==1\ndmipy-sim==3.0.0\n")
    (tmp_path / "requirements-x-zero.txt").write_text("# x was accepted on this one\ndmipy-sim==3.0.1\n")
    assert d.compose_requirements(str(tmp_path), "x.toml") == "numpy>=2\n# the stack\njax==1\n# x was accepted on this one\ndmipy-sim==3.0.1\n"
    (tmp_path / "requirements-y-zero.txt").write_text("not-listed==1\n")
    try:
        d.compose_requirements(str(tmp_path), "y.toml")
    except SystemExit as e:
        assert "not-listed" in str(e)
    else:
        raise AssertionError("a pin of a package the default does not list is refused")

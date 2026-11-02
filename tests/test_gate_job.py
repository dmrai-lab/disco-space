"""The acceptance job's script runs the gate: it sets every variable the acceptance tests are gated on, so the paid job
never ends with every test skipped."""
import importlib.util
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _gate_job():
    spec = importlib.util.spec_from_file_location("gate_job_tool", os.path.join(ROOT, "tools", "gate_job.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_job_script_sets_every_variable_the_acceptance_tests_are_gated_on():
    with open(os.path.join(ROOT, "tests", "test_acceptance.py")) as f:
        gates = re.findall(r'skipif\(\s*not os\.environ\.get\("(\w+)"\)', f.read())
    assert gates, "the acceptance tests are gated on a variable"
    script = _gate_job().SCRIPT
    for name in gates:
        assert re.search(rf"(^|\s){name}=1(\s|$)", script), f"the job script does not set {name}=1"

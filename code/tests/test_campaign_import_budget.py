"""Idle proof controllers must not reserve memory for model fitting libraries."""
import json,os,subprocess,sys


def test_idle_supervisor_does_not_load_training_stack():
    command=[sys.executable,'-c',
        "import sys,json; import campaigns.autonomous_referee; "
        "print(json.dumps(sorted(set(sys.modules)&{'numpy','pandas','sklearn','scipy','torch'})))"]
    loaded=json.loads(subprocess.check_output(command,text=True,env=dict(os.environ)))
    assert loaded==[],f'Idle supervision imported memory-heavy training libraries: {loaded}'

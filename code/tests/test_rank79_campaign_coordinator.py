import ast,json,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import pytest
import psutil
from campaigns import queue_rank79_replication as queue
from campaigns import main_rank79 as main
from commitjl.pilot_io import save


class Child:
    pid=42
    def __init__(self,command,code=None):self.command=command;self.code=code
    def cmdline(self):return self.command
    def create_time(self):return 123.
    def wait(self):return self.code


def test_actual_summary_constructions_preserve_profile_and_identify_eleven_variants():
    queue_ast=ast.parse(Path(queue.__file__).read_text())
    main_ast=ast.parse(Path(main.__file__).read_text())
    calls=lambda tree:[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dict' and any(k.arg=='variant' for k in n.keywords)]
    queue_calls=calls(queue_ast);main_calls=calls(main_ast)
    assert len(queue_calls)==2 and len(main_calls)==1
    receipt=dict(profile='group-compact-v3',complete=True)
    evaluate=lambda call,env:eval(compile(ast.Expression(body=call),'<actual-result-dict>','eval'),env)
    rows=[]
    for fit in queue.PROOF_FITS:
        for name in ['head79','head63','pca79']:
            env=dict(fit=fit,name=name,r=receipt,comparison=Path('unused'),read=lambda _:receipt)
            first=evaluate(queue_calls[0],env);second=evaluate(queue_calls[1],env)
            assert first==second;rows.append(first)
    for case in main.CASES:
        for profile in case['profiles']:
            rows.append(evaluate(main_calls[0],dict(case=case,profile=profile,root=Path('.'),verify_group=lambda _:receipt)))
    assert len(rows)==11 and all(r['profile']=='group-compact-v3' for r in rows)
    assert {r['variant'] for r in rows}=={'head79','head63','pca79'}


def test_live_child_adoption_requires_exact_phase_argv_and_pid_identity(monkeypatch):
    child=Child(['python','-m','approved.worker']);monkeypatch.setattr(queue.psutil,'Process',lambda _:child)
    state=dict(child_pid=42,phase='full',child_created_unix=123.)
    assert queue.adopt_saved_child(state,'full',child.command) is child
    for phase,command,created in [('other',child.command,123.),('full',['python','-m','other.worker'],123.),('full',child.command,125.)]:
        with pytest.raises(RuntimeError):queue.adopt_saved_child(dict(state,child_created_unix=created),phase,command)


def test_dead_saved_child_can_be_restarted(monkeypatch):
    def missing(_):raise psutil.NoSuchProcess(42)
    monkeypatch.setattr(queue.psutil,'Process',missing)
    assert queue.adopt_saved_child(dict(child_pid=42,phase='full'),'full',['python']) is None


@pytest.mark.parametrize('exit_code',[None,0])
def test_actual_invoke_adopts_existing_worker_without_launching_or_changing_cost(tmp_path,monkeypatch,exit_code):
    command=['python','-m','approved.worker'];child=Child(command,exit_code)
    monkeypatch.setattr(queue.psutil,'Process',lambda _:child)
    status=tmp_path/'source/status.json';save(status,dict(complete=True,status='complete',cases=dict(full=dict(complete=True))))
    costs=tmp_path/'source/full/full_reuse.resource.json';save(costs,dict(wall_seconds=777.,attempts=[dict(pid=999,complete=True)]));original=costs.read_bytes()
    state=dict(child_pid=42,phase='full',completed=[])
    def forbidden(*args,**kwargs):raise AssertionError('duplicate worker launch')
    module=ast.parse(Path(queue.__file__).read_text());function=next(n for n in ast.walk(module) if isinstance(n,ast.FunctionDef) and n.name=='invoke')
    env=dict(root=tmp_path,state=state,env={},subprocess=SimpleNamespace(Popen=forbidden),psutil=queue.psutil,time=time,os=queue.os,save=save,read=queue.read,adopt_saved_child=queue.adopt_saved_child,completed_child_status=queue.completed_child_status)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<actual-coordinator-invoke>','exec'),env)
    env['invoke'](command,'full',status,status.parent)
    assert state['completed']==['full'] and state['child_pid'] is None
    assert len(state['child_adoptions'])==1 and len(list(tmp_path.glob('full.adoption_*.json')))==1
    assert costs.read_bytes()==original


def test_completion_requires_recorded_finished_artifacts(tmp_path):
    status=tmp_path/'status.json';save(status,dict(complete=True,status='complete'))
    assert not queue.completed_child_status(status)
    save(tmp_path/'checked_comparison.json',dict(complete=True))
    assert queue.completed_child_status(status)
    save(status,dict(complete=True,status='complete',paused=True))
    assert not queue.completed_child_status(status)
    save(status,dict(complete=True,status='complete',cases=dict(full=dict(complete=False))))
    assert not queue.completed_child_status(status,tmp_path)
    save(status,dict(complete=True,status='complete',cases=dict(full=dict(complete=True))))
    assert queue.completed_child_status(status,tmp_path)

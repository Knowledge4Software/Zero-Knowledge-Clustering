import json
from pathlib import Path
import sys

import pytest

from campaigns import paper_queue as queue


def test_queue_includes_all_research_stages_and_full_balanced_design(tmp_path):
    from experiments.rq2345_complete import STAGES
    plan=queue.build_plan(Path(__file__).parents[1],tmp_path,sys.executable)
    included=set()
    for phase in plan:
        command=phase['command']
        assert phase['timeout_seconds']>0
        if '--stages' in command:
            first=command.index('--stages')+1
            last=command.index('--proof-reps')
            included.update(command[first:last])
        if '--run' in command:
            assert command[command.index('--seeds')+1]=='10'
            assert command[command.index('--starts')+1]=='5'
        assert '--proof-points' not in command
    assert included==set(STAGES)


def test_timeout_stops_a_phase_that_never_finishes(tmp_path):
    outcome=queue.run_bounded([sys.executable,'-c','import time; time.sleep(30)'],
        tmp_path,tmp_path/'phase.log',timeout_seconds=.25,poll_seconds=.05,
        min_free_gib=0)
    assert outcome['reason']=='phase_timeout'
    assert outcome['exit_code']!=0
    assert outcome['elapsed_seconds']<10


@pytest.mark.skipif(sys.platform!='linux',reason='Linux flock and service queue')
def test_failure_skips_dependents_but_continues_independent_work(tmp_path):
    def phase(name,script,requires):
        return dict(name=name,command=[sys.executable,'-c',script],
                    timeout_seconds=10,requires=requires)
    plan=[phase('failed','raise SystemExit(7)',[]),
          phase('dependent','raise SystemExit(0)',['failed']),
          phase('independent','print("done")',[])]
    assert queue.run_queue(tmp_path,tmp_path/'queue',plan)==1
    state=json.loads((tmp_path/'queue/queue_state.json').read_text())
    assert state['phases']['failed']['exit_code']==7
    assert state['phases']['dependent']['status']=='skipped'
    assert state['phases']['independent']['status']=='completed'
    before=state['phases']['independent']['finished_at_utc']
    assert queue.run_queue(tmp_path,tmp_path/'queue',plan)==1
    state=json.loads((tmp_path/'queue/queue_state.json').read_text())
    assert state['phases']['independent']['finished_at_utc']==before

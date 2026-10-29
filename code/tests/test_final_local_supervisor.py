from campaigns.final_local_supervisor import process_matches


def test_adoption_requires_the_same_root_and_timing_worker(tmp_path):
    class Process:
        def __init__(self,args):self.args=args
        def cmdline(self):return self.args
    args=['python','-m','campaigns.referee_final_local','--root',str(tmp_path),'--timing-only']
    assert process_matches(Process(args),tmp_path)
    assert not process_matches(Process(args[:-1]),tmp_path)
    assert not process_matches(Process(args),tmp_path/'different')
    assert not process_matches(Process(['python','unrelated.py']),tmp_path)

import argparse
import json
import pytest
from malow.vla.evaluate import TASKS, aggregate_results, build_plan


def args(kind='c2l', workers=4):
    return argparse.Namespace(kind=kind, workers=workers, checkpoint='/ckpt',
        stage1_checkpoint='/stage1', base_model='Qwen/Qwen3.5-2B',
        simpler_env='/SimplerEnv', output='/results', seed=42, port=10100,
        server_python='/server/python', sim_python='/sim/python')


def test_exact_reference_protocol():
    plan=build_plan(args())
    assert len(plan['servers'])==len(plan['tasks'])==4
    assert [s['env']['PYTHONHASHSEED'] for s in plan['servers']]==['42','43','44','45']
    for i,t in enumerate(plan['tasks']):
        cmd=t['argv']
        assert '--interactive-c2l' in cmd and '--c2l-action-feedback' in cmd
        assert cmd[cmd.index('--control-freq')+1]=='5'
        assert cmd[cmd.index('--sim-freq')+1]=='500'
        assert cmd[cmd.index('--max-episode-steps')+1]=='120'
        assert cmd[cmd.index('--action-ensemble-horizon')+1]=='4'
        assert cmd[cmd.index('--seed')+1]==str(42+i)
        assert cmd[cmd.index('--obj-episode-range')+1:cmd.index('--obj-episode-range')+3]==['0','24']
    assert 'widowx_sink_camera_setup' in plan['tasks'][3]['argv']


def test_baseline_has_no_stateful_flags():
    p=build_plan(args('baseline',2))
    assert len(p['servers'])==2
    assert [t['worker'] for t in p['tasks']]==[0,1,0,1]
    assert all('--stage1-checkpoint' not in s['argv'] for s in p['servers'])
    assert all('--interactive-c2l' not in t['argv'] and '--c2l-action-feedback' not in t['argv'] for t in p['tasks'])


def write_results(path, counts=(7,18,21,20)):
    for i,(name,count) in enumerate(zip(TASKS,counts)):
        (path/f'worker_{i}_{name}_results.json').write_text(json.dumps(dict(
            worker_id=i,env_name=name,success_array=[1]*count+[0]*(24-count),
            num_episodes=24,avg_success=count/24)))


def test_exact_aggregate_from_episode_counts(tmp_path):
    write_results(tmp_path)
    result=aggregate_results(tmp_path,4)
    assert result['total_episodes']==96
    assert result['overall']==66/96


def test_reject_incomplete_and_stale_results(tmp_path):
    write_results(tmp_path)
    first=next(tmp_path.glob('worker_*'))
    first.unlink()
    with pytest.raises(ValueError,match='four task'):aggregate_results(tmp_path,4)
    write_results(tmp_path)
    (tmp_path/'worker_4_extra_results.json').write_text('{}')
    with pytest.raises(ValueError,match='four task'):aggregate_results(tmp_path,4)


def test_reject_short_or_inconsistent_results(tmp_path):
    write_results(tmp_path)
    file=next(tmp_path.glob('worker_*'))
    data=json.loads(file.read_text());data['success_array'].pop();file.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='24 completed'):aggregate_results(tmp_path,4)
    write_results(tmp_path)
    data=json.loads(file.read_text());data['avg_success']=0.0;file.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='disagrees'):aggregate_results(tmp_path,4)


def test_gpu_selection_preserves_mask_and_explicit_ids(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-aa,GPU-bb')
    p=build_plan(args(workers=2))
    assert p['gpus']==['GPU-aa','GPU-bb']
    assert p['servers'][1]['env']['CUDA_VISIBLE_DEVICES']=='GPU-bb'
    assert p['tasks'][3]['env']['CUDA_VISIBLE_DEVICES']=='GPU-bb'
    a=args(workers=2);a.gpus='2,4'
    assert build_plan(a)['gpus']==['2','4']
    a.gpus='2,2'
    with pytest.raises(ValueError,match='distinct'):build_plan(a)
    a.gpus='2'
    with pytest.raises(ValueError,match='distinct'):build_plan(a)

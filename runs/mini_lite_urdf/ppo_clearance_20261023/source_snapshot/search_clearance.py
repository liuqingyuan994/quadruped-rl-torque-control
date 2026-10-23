"""Small actor-weight search with unchanged physics and frozen acceptance gates."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
from dataclasses import replace,asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import hashlib
import json
import sys
import time
import numpy as np

PROJECT=Path('D:/Download/桌面/rl')
ROOT=Path(__file__).resolve().parent
PREVIOUS=PROJECT/'results/mini_lite_urdf/ppo_curves_20261023/acceptance.json'
SOURCE=Path('E:/Codex/2026-10-19/x20-2/outputs/overview_training_analysis_20261023/ppo_evaluation_policy.zip')
sys.path.insert(0,str(PROJECT))
from purequad.numpy_policy import NumpyPolicy
from purequad.config import Config
from purequad.env import QuadrupedEnv
from purequad.evaluate import rollout

MODEL=BASE=None
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def initialize():
    global MODEL,BASE
    MODEL=NumpyPolicy(ROOT/'source_policy.zip')
    BASE={k:v.copy() for k,v in MODEL.weights.items()}

def edit(parameters):
    """Only edit RR hip/knee clock-feature weights; retain all feedback layers."""
    w=BASE['action_net.weight'].copy()
    for row,gain,phase in [(10,parameters['hip_gain'],parameters['hip_phase']),
                           (11,parameters['knee_gain'],parameters['knee_phase'])]:
        for harmonic in range(3):
            col=124+2*harmonic
            a,b=BASE['action_net.weight'][row,col:col+2].astype(float)
            angle=phase*(harmonic+1)
            w[row,col]=gain*(a*np.cos(angle)-b*np.sin(angle))
            w[row,col+1]=gain*(a*np.sin(angle)+b*np.cos(angle))
    MODEL.weights['action_net.weight']=w

def gates(case):
    failures=[]
    checks={'complete':case['success'],'duration':case['duration_s']<=60,
            'rmse':case['cross_track_rmse_m']<=.05,'cross_max':case['cross_track_max_m']<=.15,
            'tilt':case['max_tilt_rad']<=.08,'height':case['min_height_m']>=.28,
            'height_range':case['height_peak_to_peak_m']<=.008,
            'body_contact':case['nonfoot_contact_fraction']==0,
            'joint_speed':case['max_joint_speed_rad_s']<=20,
            'foot_height':min(case['feet_max_height_m'])>=.029,
            'foot_air':all(.08<=v<=.55 for v in case['foot_air_fraction'])}
    failures=[k for k,v in checks.items() if not v]
    return failures

def measure(parameters,path='straight',offset=-.05):
    edit(parameters)
    cfg=replace(Config(**MODEL.pure_config),stage='path',seconds=75,reference_reward=False,reset_noise=0)
    env=QuadrupedEnv(cfg,path)
    try:result,_=rollout(MODEL,env,{'y_offset':offset})
    finally:env.close()
    result['y_offset_m']=offset
    result['failed_criteria']=gates(result)
    result['passed']=not result['failed_criteria']
    return result

def target(job):
    i,parameters=job
    started=time.monotonic()
    result=measure(parameters)
    return {'index':i,'parameters':parameters,'case':result,'wall_s':time.monotonic()-started}

def full(job):
    i,parameters=job
    results=[measure(parameters,path,offset) for path in ['straight','sine','turn'] for offset in [0.,-.05,.05]]
    return {'index':i,'parameters':parameters,'cases':results,'all_passed':all(c['passed'] for c in results),
            'pass_count':sum(c['passed'] for c in results),
            'min_foot_peak_m':min(min(c['feet_max_height_m']) for c in results),
            'mean_rmse_m':float(np.mean([c['cross_track_rmse_m'] for c in results])),
            'mean_roll_rms_rad':float(np.mean([c['roll_rms_rad'] for c in results])),
            'mean_pitch_rms_rad':float(np.mean([c['pitch_rms_rad'] for c in results])),
            'mean_height_ptp_m':float(np.mean([c['height_peak_to_peak_m'] for c in results])),
            'mean_joint_HF_rad':float(np.mean([c['joint_high_frequency_rms_rad'] for c in results]))}

def parameter_set():
    zero={'hip_gain':1.,'knee_gain':1.,'hip_phase':0.,'knee_phase':0.}
    values=[zero]
    for key in ['hip_gain','knee_gain']:
        for delta in [-.02,-.01,-.005,.005,.01,.02]:values.append(dict(zero,**{key:1+delta}))
    for delta in [-.01,-.005,.005,.01,.02]:values.append(dict(zero,hip_gain=1+delta,knee_gain=1+delta))
    for key in ['hip_phase','knee_phase']:
        for phase in [-.01,-.005,.005,.01]:values.append(dict(zero,**{key:phase}))
    for hip_gain in [1.01,1.0125,1.015]:
        for knee_gain in [1.0075,1.01,1.0125,1.015]:
            item=dict(zero,hip_gain=hip_gain,knee_gain=knee_gain)
            if item not in values:values.append(item)
    return values

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true')
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--candidates',type=int,default=3)
    args=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    previous=json.loads(PREVIOUS.read_text(encoding='utf-8'))
    assert sha(SOURCE)==previous['sha256']
    frozen=ROOT/'source_policy.zip'
    if not frozen.exists():frozen.write_bytes(SOURCE.read_bytes())
    assert sha(frozen)==previous['sha256']
    config={'source':str(SOURCE),'source_sha256':sha(SOURCE),'method':'RR hip/knee clock-feature local weight search',
            'acceptance_criteria':previous['criteria'],'physics_dt':.002,'action_scale':.4,
            'parameters':parameter_set(),'selection_target_foot_height_m':.02925,
            'source_code_sha256':{f:sha(PROJECT/f) for f in ['purequad/env.py','purequad/validate.py','purequad/evaluate.py']}}
    (ROOT/'search_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    if not args.full:
        parameters=parameter_set();results=[]
        if (ROOT/'target_results.json').exists():
            results=json.loads((ROOT/'target_results.json').read_text(encoding='utf-8'))
            for previous_result in results:assert parameters[previous_result['index']]==previous_result['parameters']
        completed={r['index'] for r in results}
        with ProcessPoolExecutor(args.workers,initializer=initialize) as pool:
            futures=[pool.submit(target,(i,p)) for i,p in enumerate(parameters) if i not in completed]
            for future in as_completed(futures):
                result=future.result();results.append(result)
                with (ROOT/'target_candidates.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(result)+'\n')
                c=result['case']
                print(json.dumps({'candidate':result['index'],'parameters':result['parameters'],
                                  'RR_peak_mm':c['feet_max_height_m'][3]*1000,
                                  'min_peak_mm':min(c['feet_max_height_m'])*1000,
                                  'rmse_cm':c['cross_track_rmse_m']*100,'passed':c['passed'],
                                  'completed':len(results),'total':len(parameters)}),flush=True)
        (ROOT/'target_results.json').write_text(json.dumps(sorted(results,key=lambda x:x['index']),ensure_ascii=False,indent=2),encoding='utf-8')
    else:
        results=json.loads((ROOT/'target_results.json').read_text(encoding='utf-8'))
        base=next(r['case'] for r in results if r['index']==0)
        suitable=[r for r in results if r['case']['passed'] and min(r['case']['feet_max_height_m'])>=.02925
                  and r['case']['cross_track_rmse_m']<=base['cross_track_rmse_m']*1.03
                  and r['case']['roll_rms_rad']<=base['roll_rms_rad']*1.05
                  and r['case']['height_peak_to_peak_m']<=base['height_peak_to_peak_m']*1.05]
        def distance(r):
            p=r['parameters']
            return sum((p[k]-(1 if 'gain' in k else 0))**2 for k in p)
        suitable=sorted(suitable,key=distance)[:args.candidates]
        assert suitable,'No suitable candidate; inspect target results before expanding the search.'
        print(json.dumps({'full_screen_candidates':[r['index'] for r in suitable]}),flush=True)
        all_results=[]
        with ProcessPoolExecutor(min(args.workers,len(suitable)),initializer=initialize) as pool:
            futures=[pool.submit(full,(r['index'],r['parameters'])) for r in suitable]
            for future in as_completed(futures):
                result=future.result();all_results.append(result)
                with (ROOT/'full_candidates.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(result)+'\n')
                print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
        (ROOT/'full_results.json').write_text(json.dumps(all_results,ensure_ascii=False,indent=2),encoding='utf-8')
        feasible=[r for r in all_results if r['all_passed']]
        if not feasible:raise RuntimeError('Full suite screening failed; refine candidates.')
        # Prefer no aggregate tracking/smoothness degradation, then clearance margin.
        means=lambda key:float(np.mean([c[key] for c in previous['cases']]))
        feasible=[r for r in feasible if r['mean_rmse_m']<=means('cross_track_rmse_m')*1.03
                  and r['mean_roll_rms_rad']<=means('roll_rms_rad')*1.05
                  and r['mean_height_ptp_m']<=means('height_peak_to_peak_m')*1.05
                  and r['mean_joint_HF_rad']<=means('joint_high_frequency_rms_rad')*1.05]
        if not feasible:raise RuntimeError('Passing candidates degraded motion; refine candidates.')
        selected=min(feasible,key=lambda r:(max(0,.02925-r['min_foot_peak_m']),r['mean_rmse_m'],distance(r)))
        initialize();edit(selected['parameters'])
        provenance={'method':config['method'],'source_sha256':sha(SOURCE),'parameters':selected['parameters'],
                    'screened_full_suite':selected,'criteria_unchanged':True}
        MODEL.save(ROOT/'selected_policy.zip',MODEL.pure_config,provenance)
        selected['policy_sha256']=sha(ROOT/'selected_policy.zip')
        (ROOT/'selected.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'event':'selected','index':selected['index'],'sha256':selected['policy_sha256']}),flush=True)

if __name__=='__main__':main()

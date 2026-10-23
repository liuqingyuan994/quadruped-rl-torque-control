"""Frozen-model acceptance: full paths, feet lifting, stability and execution without motion rewards."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,hashlib
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .numpy_policy import NumpyPolicy
from .config import Config
from .model import MODEL_ID
from .env import QuadrupedEnv
from .evaluate import rollout,save


def case(job):
    filename,path,offset,seconds,output=job
    model=NumpyPolicy(filename);cfg=replace(Config(**model.pure_config),stage='path',seconds=seconds,reference_reward=False)
    env=QuadrupedEnv(cfg,path,visuals=True)
    try:
        r, rows = rollout(
        model,
        env,
        {'y_offset': offset},
        video=str(Path(output) / 'motion.mp4'),
        video_fps=60,
        camera_mode='stable',
        )
        r['y_offset_m'] = offset
        #r,rows=rollout(model,env,{'y_offset':offset});r['y_offset_m']=offset
        r['passed']=bool(r['success'] and r['duration_s']<=60 and r['cross_track_rmse_m']<=.05
            and r['cross_track_max_m']<=.15 and r['max_tilt_rad']<=.08 and r['min_height_m']>=.28
            and r['height_peak_to_peak_m']<=.008 and r['nonfoot_contact_fraction']==0
            and r['max_joint_speed_rad_s']<=cfg.joint_speed_limit_rad_s
            and min(r['feet_max_height_m'])>=env.foot_radius+cfg.foot_clearance_min_m and all(.08<=a<=.55 for a in r['foot_air_fraction']))
        if output:save(output,r,rows)
        return r
    finally:env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='runs/mini_lite_urdf/final/policy.zip')
    p.add_argument('--workers',type=int,default=1);p.add_argument('--seconds',type=float,default=75)
    p.add_argument('--output',default='results/mini_lite_urdf/acceptance');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);frozen=out/'evaluated_policy.zip';frozen.write_bytes(Path(args.model).read_bytes())
    jobs=[(str(frozen),path,offset,args.seconds,str(out/f'{path}_{offset:+.2f}')) for path in ['straight','sine','turn'] for offset in [0.,-.05,.05]]
    if args.workers==1:
        cases=[]
        for job in jobs:
            result=case(job);cases.append(result)
            print(result['path'],result['y_offset_m'],result['reason'],result['passed'],flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:cases=list(ex.map(case,jobs))
    report={'model':args.model,'sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),'all_passed':all(c['passed'] for c in cases),
        'pass_rate':float(np.mean([c['passed'] for c in cases])),'reference_reward_enabled':False,
        'criteria':{'complete':True,'duration_s_max':60,'rmse_m_max':.05,'cross_max_m':.15,'tilt_rad_max':.08,
            'min_height_m':.28,'height_peak_to_peak_m_max':.008,'nonfoot_contact_fraction':0,
            'all_feet_peak_height_m_min':.029,'joint_speed_max_rad_s':20.,'robot_model':MODEL_ID,'all_feet_air_fraction_range':[.08,.55]},'cases':cases}
    (out/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS' if report['all_passed'] else 'FAIL',sum(c['passed'] for c in cases),'/9',[(c['path'],c['y_offset_m'],c['reason'],round(c['cross_track_rmse_m'],3),round(c['duration_s'],1)) for c in cases],flush=True)


if __name__=='__main__':main()

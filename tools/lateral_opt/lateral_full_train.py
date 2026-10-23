"""Local reward search evaluated on three complete paths, with frozen physical limits."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from dataclasses import asdict,replace
from concurrent.futures import ProcessPoolExecutor
import json,argparse,time
import numpy as np
import lateral_full_measure as measure
from purequad.numpy_policy import NumpyPolicy
from purequad.config import Config
BASE=REFERENCE=None
def weights(base,p):
    w=base.copy();p=np.clip(np.asarray(p),-.4,.4)
    for leg in range(4):
        side=1 if leg in (0,2) else -1;front=1 if leg<2 else -1
        w[3*leg,124:130]*=np.exp(p[0])
        w[3*leg+1,124:130]*=np.exp(front*p[1])
        w[3*leg+2,124:130]*=np.exp(front*p[2]+p[12])
        w[3*leg,68]+=side*p[8]
        for j in range(3):w[3*leg+j,85+3*leg+j]*=np.exp(p[9+j])
    w[:,65]*=np.exp(p[3]);w[:,70]*=np.exp(p[4]);w[:,72]*=np.exp(p[5])
    w[:,120]*=np.exp(p[6]);w[:,137:143]*=np.exp(p[6])
    w[:,119]*=np.exp(p[7]);w[:,131:137]*=np.exp(p[7]);w[:,128:130]*=np.exp(p[13])
    return w
def initialize(source,cfg,cases,reference):
    global BASE,REFERENCE
    measure.initialize(source,cfg,cases,reference);BASE=measure.BASE.copy();REFERENCE=reference
def candidate(p):
    measure.BASE=weights(BASE,p);r=measure.candidate(np.zeros(71));cases=r['cases']
    keys=['roll_std_rad','yaw_gait_rms_rad','lateral_gait_rms_m','HF10_rad','acceleration_rms_rad_s2',
          'torque_rate_rms_Nm_s','height_peak_to_peak_m','cross_track_rmse_m','heading_std_rad','pitch_std_rad']
    ratios={k:float(np.mean([c[k] for c in cases])/np.mean([c[k] for c in REFERENCE])) for k in keys}
    failures=[]
    for i,(c,b) in enumerate(zip(cases,REFERENCE)):
        valid=c['reason']=='success' and c['duration_s']<=min(60,1.05*b['duration_s'])
        valid=valid and c['cross_track_rmse_m']<=.05 and c['cross_track_max_m']<=.15
        valid=valid and c['min_height_m']>=.28 and c['max_tilt_rad']<=.08 and c['height_peak_to_peak_m']<=.008
        valid=valid and c['nonfoot_contact_fraction']==0 and c['max_joint_speed_rad_s']<=20
        valid=valid and min(c['feet_max_height_m'])>=.029 and all(.08<=x<=.55 for x in c['air_fraction'])
        if not valid:failures.append(i)
    excess=sum(max(0,ratios[k]-1.03) for k in ['roll_std_rad','yaw_gait_rms_rad','lateral_gait_rms_m'])
    excess+=max(0,ratios['HF10_rad']-1.25)+max(0,ratios['heading_std_rad']-1.08)
    feasible=not failures and excess<1e-10
    cost=2500*sum(ratios[k]**2 for k in ['roll_std_rad','yaw_gait_rms_rad','lateral_gait_rms_m'])
    cost+=1500*sum(ratios[k]**2 for k in ['HF10_rad','acceleration_rms_rad_s2','torque_rate_rms_Nm_s'])
    cost+=1000*(ratios['height_peak_to_peak_m']**2+ratios['cross_track_rmse_m']**2)+700*ratios['pitch_std_rad']**2
    score=-float(cost) if feasible else -1e6-1e5*(len(failures)+excess)
    return {'return':score,'feasible':feasible,'ratios':ratios,'failed_center_cases':failures,'cases':cases,'steps':r['steps']}
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--cases',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--population',type=int,default=18);p.add_argument('--generations',type=int,default=4);p.add_argument('--seed',type=int,default=1021)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);assert not any(out.iterdir())
    model=NumpyPolicy(args.source);base=model.weights['action_net.weight'].copy();cfg=replace(Config(**model.pure_config),seconds=60,reference_reward=False)
    cases=json.loads(Path(args.cases).read_text(encoding='utf-8'));reference=json.loads(Path(args.reference).read_text(encoding='utf-8'))
    (out/'config.json').write_text(json.dumps({'args':vars(args),'cases':cases,'reference':reference,'environment':asdict(cfg),
        'reward':'normalized full-path lateral/yaw/roll, joint vibration and tracking objectives; same frozen physical gates'},indent=2),encoding='utf-8')
    std=np.array([.10,.025,.025,.08,.10,.12,.08,.08,.14,.06,.06,.06,.025,.08]);floor=std*.08
    mean=np.zeros(14);bestp=mean.copy();best=-np.inf;steps=episodes=0;rng=np.random.default_rng(args.seed)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(args.source,asdict(cfg),cases,reference)) as pool:
        for generation in range(args.generations):
            parameters=rng.normal(mean,std,(args.population,14));parameters[0]=bestp
            if generation==0:parameters[1]=0
            start=time.perf_counter();rows=list(pool.map(candidate,parameters));scores=np.array([x['return'] for x in rows]);i=int(scores.argmax())
            steps+=sum(r['steps'] for r in rows);episodes+=args.population*len(cases)
            if scores[i]>best:
                best=float(scores[i]);bestp=parameters[i].copy();model.weights['action_net.weight']=weights(base,bestp)
                meta={'source':args.source,'method':'complete_path_lateral_joint_reward_CEM','generation':generation,'episodes':episodes,'environment_steps':steps,'parameters':bestp.tolist(),'metrics':rows[i]}
                model.save(out/'best_policy.zip',asdict(cfg),meta);model.save(out/f'generation_{generation:03d}.zip',asdict(cfg),meta)
                (out/'best_candidate.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
            with (out/'generations.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'generation':generation,'parameters':parameters.tolist(),'candidates':rows})+'\n')
            (out/'training_state.json').write_text(json.dumps({'generation':generation,'episodes':episodes,'steps':steps,'best_return':best,'active':True},indent=2),encoding='utf-8')
            print(json.dumps({'generation':generation,'best':round(best,1),'feasible_count':sum(x['feasible'] for x in rows),'ratios':rows[i]['ratios'],'seconds':round(time.perf_counter()-start,1)}),flush=True)
            elite=parameters[np.argsort(scores)[-max(4,args.population//4):]];mean=.2*mean+.8*elite.mean(axis=0);std=np.maximum(.65*std+.35*elite.std(axis=0),floor)
            if (out/'STOP').exists():break
    (out/'training_state.json').write_text(json.dumps({'generation':generation,'episodes':episodes,'steps':steps,'best_return':best,'active':False},indent=2),encoding='utf-8')
if __name__=='__main__':main()

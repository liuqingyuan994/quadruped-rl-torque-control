"""Reward-based direct-policy refinement targeting lateral sway and yaw rocking."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from dataclasses import replace,asdict
from concurrent.futures import ProcessPoolExecutor
import sys,json,time,argparse,itertools
import numpy as np
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from purequad.numpy_policy import NumpyPolicy
from purequad.env import QuadrupedEnv
from purequad.config import Config
from purequad.paths import Path as Track
from purequad.metrics import high_frequency_rms
from jitter_body_yaw_train import weights as body_weights
POLICY=ENV=BASE=CASES=LIMITS=None
DEFAULT_SECONDS=8

def weights(p,base):
    p=np.asarray(p);w=body_weights(p[:57],base)
    cross,head,yaw,roll,rollrate,side=np.exp(np.clip(p[57:63],-1.8,1.8))
    w[:,119]*=cross;w[:,131:137]*=cross
    w[:,120]*=head;w[:,137:143]*=head;w[:,72]*=yaw
    w[:,65]*=roll;w[:,70]*=rollrate;w[:,68]*=side
    for leg in range(4):
        side_sign=1 if leg in (0,2) else -1
        for joint in range(3):
            row=3*leg+joint
            w[row,130]+=np.clip(p[63+joint],-.12,.12)
            for h in range(3):
                c=124+2*h;angle=side_sign*(h+1)*np.clip(p[66+joint],-.10,.10)
                a,b=w[row,c:c+2].copy()
                w[row,c]=a*np.cos(angle)-b*np.sin(angle)
                w[row,c+1]=a*np.sin(angle)+b*np.cos(angle)
    for leg in range(4):w[3*leg,124:130]*=np.exp(np.clip(p[69],-1.6,.6))
    for leg in range(4):w[3*leg+2,124:130]*=np.exp(np.clip(p[70],-.30,.30))
    return w

def initialize(source,cfg,cases,limits):
    global POLICY,ENV,BASE,CASES,LIMITS,DEFAULT_SECONDS
    POLICY=NumpyPolicy(source);BASE=POLICY.weights['action_net.weight'].copy()
    ENV=QuadrupedEnv(Config(**cfg));CASES=cases;LIMITS=limits;DEFAULT_SECONDS=cfg['seconds']

def candidate(parameters):
    POLICY.weights['action_net.weight']=weights(parameters,BASE)
    results=[]
    for case_index,options in enumerate(CASES):
        ENV.path_name=options['path'];ENV.path=Track(options['path']);ENV.cfg.seconds=float(options.get('seconds',DEFAULT_SECONDS))
        obs,_=ENV.reset(seed=0,options=options);begin=ENV.last_progress;done=False;total=0.
        count=0;accel=rate=0.;previous=None;q=[];body=[];feet=np.zeros(4);air=np.zeros(4)
        max_tilt=bad=0;cross=0.;crossmax=maxspeed=0.;minimum_height=1.;zmin=1.;zmax=0.
        while not done:
            action=POLICY.predict(obs)[0];obs,r,d,t,info=ENV.step(action);done=d or t;total+=r
            cross+=info['cross_track_m']**2;bad+=info['nonfoot_contacts']>0
            crossmax=max(crossmax,abs(info['cross_track_m']));maxspeed=max(maxspeed,max(abs(v) for v in info['joint_velocity_rad_s']));minimum_height=min(minimum_height,info['position'][2])
            max_tilt=max(max_tilt,max(abs(x) for x in info['rpy'][:2]))
            if info['time_s']>=1.:
                feet=np.maximum(feet,info['feet_z']);air+=1-np.array(info['contacts']);count+=1
                zmin=min(zmin,info['position'][2]);zmax=max(zmax,info['position'][2])
            if info['time_s']>=4.:
                q.append(info['joint_position_rad'])
                body.append([*info['rpy'][:2],info['heading_error_rad'],info['cross_track_m']])
                accel+=np.mean(np.square(info['joint_acceleration_rad_s2']))
                torque=np.array(info['torque_Nm'])
                if previous is not None:rate+=np.mean(((torque-previous)/ENV.dt)**2)
                previous=torque
        n=len(q);progress=info['progress_m']-begin;height=max(0,zmax-zmin);fractions=air/max(1,count)
        if n>=8:
            b=np.asarray(body);q=np.asarray(q)
            yaw_rms=float(high_frequency_rms(b[:,2:3],ENV.dt,cutoff=.7)[0])
            side_rms=float(high_frequency_rms(b[:,3:4],ENV.dt,cutoff=.7)[0])
            roll_rms=float(np.std(b[:,0]));pitch_rms=float(np.std(b[:,1]))
            hf=float(np.sqrt(np.mean(high_frequency_rms(q,ENV.dt)**2)))
            hf4=float(np.sqrt(np.mean(high_frequency_rms(q,ENV.dt,cutoff=4.)**2)))
            roll_rate=float(np.sqrt(np.mean(np.diff(b[:,0])**2)))/ENV.dt
        else:yaw_rms=side_rms=roll_rms=pitch_rms=hf=hf4=roll_rate=1.
        acc2=accel/max(1,n);rate2=rate/max(1,n-1);rmse=np.sqrt(cross/max(1,ENV.steps))
        target=min(max(0,ENV.path.total-begin-.1),.118*max(0,ENV.cfg.seconds-ENV.cfg.clock_delay))
        penalty=7000*max(0,target-progress)+50000*max(0,height-.008)
        penalty+=250000*float(np.maximum(.030-feet,0).sum())
        penalty+=1000*float((np.maximum(.08-fractions,0)+np.maximum(fractions-.55,0)).sum())
        penalty+=5000*max(0,max_tilt-.08)+3000*bad/max(1,ENV.steps)+2000*rmse
        if info['reason'] not in ('timeout','success'):penalty+=10000+500*max(0,ENV.cfg.seconds-info['time_s'])
        reference=LIMITS[case_index]
        penalty+=3e5*max(0,roll_rms-reference['roll_std_rad'])
        penalty+=1e6*max(0,side_rms-reference['lateral_gait_rms_m'])
        penalty+=1e5*max(0,yaw_rms-reference['yaw_gait_rms_rad'])
        smooth=1.5e6*yaw_rms**2+1.5e8*side_rms**2+1e7*roll_rms**2+1e6*pitch_rms**2
        smooth+=4e6*hf**2+1e6*hf4**2+.002*acc2+.002*rate2+4e6*height**2+30*roll_rate**2
        score=total-penalty-smooth
        results.append({'path':options['path'],'return':float(score),'raw_return':total,'constraint_penalty':float(penalty),
            'yaw_gait_rms_rad':yaw_rms,'lateral_gait_rms_m':side_rms,'roll_std_rad':roll_rms,'pitch_std_rad':pitch_rms,
            'heading_std_rad':float(np.std(b[:,2])) if n>=8 else 1.,'heading_rms_rad':float(np.sqrt(np.mean(b[:,2]**2))) if n>=8 else 1.,'HF10_rad':hf,'HF4_rad':hf4,'acceleration_rms_rad_s2':float(np.sqrt(acc2)),
            'torque_rate_rms_Nm_s':float(np.sqrt(rate2)),'roll_rate_rms_rad_s':roll_rate,
            'progress_m':progress,'duration_s':info['time_s'],'reason':info['reason'],
            'height_peak_to_peak_m':height,'feet_max_height_m':feet.tolist(),'air_fraction':fractions.tolist(),
            'cross_track_rmse_m':float(rmse),'cross_track_max_m':crossmax,'min_height_m':minimum_height,'max_tilt_rad':max_tilt,'max_joint_speed_rad_s':maxspeed,'nonfoot_contact_fraction':bad/max(1,ENV.steps),'steps':ENV.steps})
    values=[x['return'] for x in results]
    ratios={key:float(np.mean([x[key] for x in results])/np.mean([x[key] for x in LIMITS])) for key in ['yaw_gait_rms_rad','lateral_gait_rms_m','roll_std_rad']}
    speed_deficit=sum(max(0,.97*ref['progress_m']-x['progress_m']) for x,ref in zip(results,LIMITS))
    feasible=speed_deficit<1e-10 and all(v<=1.03 for v in ratios.values())
    value=float(.5*min(values)+.5*np.mean(values))
    if not feasible:value=-1e6-1e5*(speed_deficit+sum(max(0,v-1.03) for v in ratios.values()))
    return {'return':value,'steps':sum(x['steps'] for x in results),'cases':results,'feasible':feasible,'relative_sway':ratios,'progress_deficit_m':speed_deficit}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--cases',required=True)
    p.add_argument('--limits',required=True)
    p.add_argument('--mode',choices=['grid','cem'],default='cem');p.add_argument('--workers',type=int,default=6)
    p.add_argument('--seconds',type=float,default=8);p.add_argument('--population',type=int,default=36)
    p.add_argument('--generations',type=int,default=10);p.add_argument('--sigma',type=float,default=.35);p.add_argument('--seed',type=int,default=515)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);assert not any(out.iterdir())
    model=NumpyPolicy(args.source);base=model.weights['action_net.weight'].copy()
    np.testing.assert_array_equal(weights(np.zeros(71),base),base)
    cfg=replace(Config(**model.pure_config),seconds=args.seconds,reference_reward=False)
    cases=json.loads(Path(args.cases).read_text(encoding='utf-8'));limits=json.loads(Path(args.limits).read_text(encoding='utf-8'))
    (out/'config.json').write_text(json.dumps({'args':vars(args),'environment':asdict(cfg),'cases':cases,'limits':limits,
        'objective':'explicit >=0.7Hz heading/lateral sway and roll/pitch variability with locomotion, lift and smoothness constraints',
        'control':'same single learned 12 full torques; unchanged camera, physical model and acceptance'},indent=2),encoding='utf-8')
    std=np.r_[[.18,.40,.22],np.full(3,.13),np.tile([.06,.10,.09],3),np.tile([.08,.12,.12],2),np.tile([.04,.06,.06],2),
        [.04,.14,.14],[.04,.14,.14],np.full(9,.06),[.16,.18,.18],np.full(9,.035),np.full(3,.035),[.15,.35,.55,.35,.55,.4]]*args.sigma
    std=np.r_[std,np.array([.04,.025,.025,.04,.04,.04])*args.sigma]
    std[:]=0.
    std[[0,3]]=[.10,.06]
    std[6:15]=[.15,.015,.015,.12,.025,.025,.10,.025,.025]
    std[15:33]=[.04,.05,.05,.02,.02,.02,.04,.08,.08,.01,.02,.02,.01,.02,.02,.01,.02,.02]
    std[33:42]=np.tile([.01,.015,.015],3)
    std[42:45]=[.10,.05,.05];std[45:54]=.015;std[54:57]=[.03,.003,.003]
    std[57:63]=[.03,.12,.22,.12,.22,.16];std[63:69]=[.02,.005,.005,.02,.01,.01]
    std*=args.sigma
    initial_floor=std*.06
    std=np.r_[std,.4*args.sigma,.06*args.sigma]
    initial_floor=std*.06
    assert len(std)==71
    rng=np.random.default_rng(args.seed);mean=np.zeros(71);bestp=mean.copy();best=-np.inf;steps=episodes=0
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(args.source,asdict(cfg),cases,limits)) as pool:
        for generation in range(1 if args.mode=='grid' else args.generations):
            if args.mode=='grid':
                parameters=[]
                for head,yaw in itertools.product([.6,.9,1.,1.2,1.6],[.6,1.,1.6,2.6,4.]):
                    a=np.zeros(71);a[58]=np.log(head);a[59]=np.log(yaw);parameters.append(a)
                parameters=np.array(parameters)
            else:
                parameters=rng.normal(mean,std,(args.population,71));parameters[0]=bestp
                if generation==0:parameters[1]=0.
            start=time.perf_counter();rows=list(pool.map(candidate,parameters));scores=np.array([x['return'] for x in rows]);k=int(scores.argmax())
            steps+=sum(x['steps'] for x in rows);episodes+=len(parameters)*len(cases)
            if scores[k]>best:
                best=float(scores[k]);bestp=parameters[k].copy();model.weights['action_net.weight']=weights(bestp,base)
                meta={'source':args.source,'method':'lateral_sway_reward_'+args.mode,'generation':generation,'episodes':episodes,
                    'environment_steps':steps,'parameters':bestp.tolist(),'metrics':rows[k]}
                model.save(out/'best_policy.zip',asdict(cfg),meta);model.save(out/f'generation_{generation:03d}.zip',asdict(cfg),meta)
                (out/'best_candidate.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
            if args.mode=='grid':
                for rank,i in enumerate(np.argsort(scores)[-5:][::-1]):
                    model.weights['action_net.weight']=weights(parameters[i],base)
                    model.save(out/f'rank_{rank}.zip',asdict(cfg),{'source':args.source,'method':'lateral_sway_reward_grid','parameters':parameters[i].tolist(),'metrics':rows[i]})
            (out/'training_state.json').write_text(json.dumps({'generation':generation,'episodes':episodes,'steps':steps,'best_return':best,'active':True},indent=2),encoding='utf-8')
            with (out/'generations.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'generation':generation,'parameters':parameters.tolist(),'candidates':rows})+'\n')
            print(json.dumps({'generation':generation,'best':round(best,1),'yaw_mrad':[round(x['yaw_gait_rms_rad']*1000,1) for x in rows[k]['cases']],
                'roll_mrad':[round(x['roll_std_rad']*1000,2) for x in rows[k]['cases']],
                'feasible_count':sum(x['feasible'] for x in rows),'relative_sway':rows[k]['relative_sway'],
                'side_mm':[round(x['lateral_gait_rms_m']*1000,2) for x in rows[k]['cases']],
                'progress':[round(x['progress_m'],3) for x in rows[k]['cases']],'seconds':round(time.perf_counter()-start,1)}),flush=True)
            elite=parameters[np.argsort(scores)[-max(6,len(parameters)//6):]];mean=.2*mean+.8*elite.mean(axis=0)
            std=np.maximum(.65*std+.35*elite.std(axis=0),initial_floor)
            if (out/'STOP').exists():break
    (out/'training_state.json').write_text(json.dumps({'generation':generation,'episodes':episodes,'steps':steps,'best_return':best,'active':False},indent=2),encoding='utf-8')
if __name__=='__main__':main()

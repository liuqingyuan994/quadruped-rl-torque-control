"""Reward-only direct-policy refinement with an explicit vibration objective."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,sys
from pathlib import Path
from dataclasses import asdict,replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from purequad.numpy_policy import NumpyPolicy
from purequad.env import QuadrupedEnv
from purequad.config import Config
from purequad.paths import Path as Track
from purequad.smooth_train import parameter_weights as smooth_weights
from purequad.metrics import high_frequency_rms

POLICY=ENV=BASE=CASES=None

def weights(parameters,base):
    p=np.asarray(parameters)
    w=smooth_weights(p[:33],base)
    for leg in range(4):
        for joint in range(3):
            row=leg*3+joint
            for h in range(3):
                c=64+60+2*h
                a,b=w[row,c:c+2].copy();theta=np.clip(p[33+3*h+joint],-.3,.3)
                w[row,c]=a*np.cos(theta)-b*np.sin(theta)
                w[row,c+1]=a*np.sin(theta)+b*np.cos(theta)
    for leg in range(4):
        front=1 if leg in (0,1) else -1
        for joint in range(3):
            scale=np.exp(np.clip(front*p[42+joint],-.4,.4))
            w[leg*3+joint,124:130]*=scale
    for leg in range(4):
        front=1 if leg in (0,1) else -1
        for joint in range(3):
            row=3*leg+joint
            for h in range(3):
                c=124+2*h;theta=np.clip(front*p[45+3*h+joint],-.20,.20)
                a,b=w[row,c:c+2].copy()
                w[row,c]=a*np.cos(theta)-b*np.sin(theta)
                w[row,c+1]=a*np.sin(theta)+b*np.cos(theta)
            w[row,:64]*=np.exp(np.clip(p[54+joint],-.2,.2))
    return w

def initialize(filename,cfg,cases):
    global POLICY,ENV,BASE,CASES
    POLICY=NumpyPolicy(filename);BASE=POLICY.weights['action_net.weight'].copy()
    ENV=QuadrupedEnv(Config(**cfg));CASES=cases

def candidate(parameters):
    POLICY.weights['action_net.weight']=weights(parameters,BASE)
    results=[]
    for options in CASES:
        ENV.path_name=options['path'];ENV.path=Track(options['path'])
        obs,_=ENV.reset(seed=0,options=options);begin=ENV.last_progress;done=False;total=0
        q=[];body=[];accel=rate=0.;prev_torque=None;feet=np.zeros(4);air=np.zeros(4);zmin=1.;zmax=0.;count=0
        max_tilt=0.;body_contacts=0;cross=0.;last_z=ENV.data.site_xpos[ENV.sites,2].copy();last_contact=np.ones(4,dtype=bool);impacts=[]
        while not done:
            action=POLICY.predict(obs)[0];obs,r,d,t,info=ENV.step(action);done=d or t;total+=r
            max_tilt=max(max_tilt,max(abs(x) for x in info['rpy'][:2]));body_contacts+=info['nonfoot_contacts']>0
            cross+=info['cross_track_m']**2
            z=np.array(info['feet_z']);contacts=np.array(info['contacts'],dtype=bool)
            if info['time_s']>=1.:
                count+=1;q.append(info['joint_position_rad']);body.append([info['position'][2],*info['rpy'][:2],info['heading_error_rad']]);accel+=np.mean(np.square(info['joint_acceleration_rad_s2']))
                torque=np.array(info['torque_Nm'])
                if prev_torque is not None:rate+=np.mean(((torque-prev_torque)/ENV.dt)**2)
                prev_torque=torque;feet=np.maximum(feet,z);air+=1-contacts
                zmin=min(zmin,info['position'][2]);zmax=max(zmax,info['position'][2])
                touchdown=contacts&~last_contact
                impacts.extend(np.maximum(0,-(z-last_z)[touchdown]/ENV.dt).tolist())
            last_z=z;last_contact=contacts
        progress=info['progress_m']-begin
        hf=high_frequency_rms(np.array(q),ENV.dt) if count>=8 else np.zeros(12)
        hf2=float(np.mean(hf**2));acc2=accel/max(1,count);rate2=rate/max(1,count)
        height=max(0,zmax-zmin);fractions=air/max(1,count)
        constraints=30000*float(np.maximum(ENV.foot_radius+ENV.cfg.foot_clearance_min_m-feet,0).sum())
        constraints+=300*float((np.maximum(.08-fractions,0)+np.maximum(fractions-.55,0)).sum())
        target=min(max(0,ENV.path.total-begin-.1),.115*max(0,ENV.cfg.seconds-ENV.cfg.clock_delay))
        constraints+=5000*max(0,target-progress)
        constraints+=50000*max(0,height-.008)+5000*max(0,max_tilt-.08)
        constraints+=2000*body_contacts/max(1,ENV.steps)
        if info['reason'] not in ('timeout','success'):constraints+=2000+200*max(0,ENV.cfg.seconds-info['time_s'])
        impact2=float(np.mean(np.square(impacts))) if impacts else 0
        body_array=np.array(body)
        body_rms=np.sqrt(np.mean(np.var(body_array[:,1:3],axis=0))) if count>1 else 0.
        vertical_rate=np.sqrt(np.mean(np.diff(body_array[:,0])**2))/ENV.dt if count>1 else 0.
        cross_rmse=float(np.sqrt(cross/max(1,ENV.steps)))
        hf4=high_frequency_rms(np.array(q),ENV.dt,cutoff=4.) if count>=8 else np.zeros(12)
        hf4_2=float(np.mean(hf4**2))
        heading_osc=float(np.var(body_array[:,3])) if count>1 else 0.
        smooth_cost=4e6*hf2+1e6*hf4_2+.002*acc2+.002*rate2+100*impact2+4e6*height**2+2e5*body_rms**2+1e5*heading_osc+400*vertical_rate**2
        constraints+=2000*cross_rmse+30000*max(0,height-.006)
        constraints+=30000*float(np.maximum(.030-feet,0).sum())
        score=total-constraints-smooth_cost
        results.append({'path':options['path'],'return':score,'raw_return':total,'constraint_penalty':constraints,'smooth_cost':smooth_cost,
                        'reason':info['reason'],'duration_s':info['time_s'],'progress_m':progress,
                        'cross_track_rmse_m':float(np.sqrt(cross/max(1,ENV.steps))),
                        'high_frequency_rms_rad':float(np.sqrt(hf2)),'joint_acceleration_rms_rad_s2':float(np.sqrt(acc2)),
                        'torque_rate_rms_Nm_s':float(np.sqrt(rate2)),'touchdown_speed_rms_m_s':float(np.sqrt(impact2)),
                        'joint_above4Hz_rms_rad':float(np.sqrt(hf4_2)),'heading_oscillation_rms_rad':float(np.sqrt(heading_osc)),'body_angle_rms_rad':float(body_rms),'vertical_velocity_rms_m_s':float(vertical_rate),'height_peak_to_peak_m':height,'max_tilt_rad':max_tilt,'feet_max_height_m':feet.tolist(),
                        'air_fraction':fractions.tolist(),'steps':ENV.steps})
    return {'return':float(min(x['return'] for x in results)),'steps':sum(x['steps'] for x in results),'cases':results}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cases',required=True);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--population',type=int,default=48);p.add_argument('--generations',type=int,default=16)
    p.add_argument('--seconds',type=float,default=6);p.add_argument('--sigma',type=float,default=1)
    p.add_argument('--seed',type=int,default=219);args=p.parse_args()
    out=Path(args.output);assert not out.exists() or not any(out.iterdir());out.mkdir(parents=True,exist_ok=True)
    model=NumpyPolicy(args.source);base=model.weights['action_net.weight'].copy()
    cfg=replace(Config(**model.pure_config),stage='path',seconds=args.seconds,reference_reward=False,reward_version=5)
    cases=json.loads(Path(args.cases).read_text(encoding='utf-8'))
    std=np.r_[[.18,.40,.22],np.full(3,.13),np.tile([.06,.10,.09],3),
              np.tile([.08,.12,.12],2),np.tile([.04,.06,.06],2),[.04,.14,.14],[.04,.14,.14],np.full(9,.06)]*args.sigma
    std=np.r_[std,np.array([.16,.18,.18])*args.sigma,np.full(9,.035)*args.sigma,np.full(3,.035)*args.sigma]
    assert len(std)==57
    floor=std*.06;mean=np.zeros(57);bestp=mean.copy();rng=np.random.default_rng(args.seed);best=-np.inf;steps=episodes=0
    configuration={'args':vars(args),'environment':asdict(cfg),'cases':cases,
                   'objective':'worst-case reward minus >=4 Hz and >=10 Hz joint vibration, body/heading oscillation, acceleration, torque-rate and landing-impact costs',
                   'runtime':'one learned direct-torque policy; physical model and acceptance criteria unchanged'}
    (out/'config.json').write_text(json.dumps(configuration,indent=2),encoding='utf-8')
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(args.source,asdict(cfg),cases)) as pool:
        for generation in range(args.generations):
            parameters=rng.normal(mean,std,(args.population,len(mean)));parameters[0]=bestp
            if generation==0:parameters[1]=np.zeros(len(mean))
            start=time.perf_counter();rows=list(pool.map(candidate,parameters));scores=np.array([r['return'] for r in rows]);k=int(scores.argmax())
            steps+=sum(x['steps'] for x in rows);episodes+=len(cases)*args.population
            if scores[k]>best:
                best=float(scores[k]);bestp=parameters[k].copy();model.weights['action_net.weight']=weights(bestp,base)
                provenance={'source':args.source,'generation':generation,'episodes':episodes,'environment_steps':steps,
                            'return':best,'parameters':bestp.tolist(),'metrics':rows[k],'purpose':'reduce body bounce and joint vibration while preserving locomotion'}
                model.save(out/'best_policy.zip',asdict(cfg),provenance);model.save(out/f'generation_{generation:03d}.zip',asdict(cfg),provenance)
                (out/'best_candidate.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
            (out/'training_state.json').write_text(json.dumps({'generation':generation,'episodes':episodes,'environment_steps':steps,'best_return':best,'active':True},indent=2),encoding='utf-8')
            with (out/'generations.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'generation':generation,'best':best,'parameters':parameters.tolist(),'candidates':rows})+'\n')
            print(json.dumps({'generation':generation,'best':round(best,2),'mean':round(float(scores.mean()),2),
                              'height_mm':[round(x['height_peak_to_peak_m']*1000,2) for x in rows[k]['cases']],
                              'HF_mrad':[round(x['high_frequency_rms_rad']*1000,2) for x in rows[k]['cases']],
                              'accel':[round(x['joint_acceleration_rms_rad_s2'],1) for x in rows[k]['cases']],
                              'progress':[round(x['progress_m'],3) for x in rows[k]['cases']],
                              'elapsed_s':round(time.perf_counter()-start,1)}),flush=True)
            elite=parameters[np.argsort(scores)[-max(8,args.population//6):]]
            mean=.2*mean+.8*elite.mean(axis=0);std=np.maximum(.65*std+.35*elite.std(axis=0),floor)
            if (out/'STOP').exists():break
    (out/'training_state.json').write_text(json.dumps({'generation':generation,'episodes':episodes,'environment_steps':steps,'best_return':best,'active':False},indent=2),encoding='utf-8')

if __name__=='__main__':main()

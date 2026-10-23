"""Reward-based local CEM refinement of an existing full-torque neural policy."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time
from dataclasses import asdict,replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from .numpy_policy import NumpyPolicy
from .cem import SEARCH_AXIS_SIGNS
import numpy as np
from .config import Config
from .env import QuadrupedEnv
from .paths import Path as TrackPath

POLICY=None
ENV=None
BASE=None
CASES=None
HEIGHT_PENALTY=10000.
OBJECTIVE='mean'


def parameter_weights(parameters,base):
    w=base.copy();p=np.asarray(parameters)
    for leg in range(4):
        side=1 if leg in (0,2) else -1
        front=1 if leg in (0,1) else -1
        for joint in range(3):
            row=leg*3+joint
            w[row,64+21+row]+=p[joint]
            w[row,64+9+row]*=np.exp(np.clip(p[3+joint],-.35,.35))
            for harmonic in range(3):
                column=64+60+harmonic*2
                w[row,column:column+2]*=np.exp(np.clip(p[6+3*harmonic+joint],-.5,.5))
            patterns=[1 if joint==0 else side,front,1 if joint==0 else side,front,side if joint==0 else 1,side if joint==0 else 1]
            for group,(column,pattern) in enumerate(zip([1,0,6,7,58,5],patterns)):
                w[row,64+column]+=pattern*(SEARCH_AXIS_SIGNS[leg] if joint==0 else 1)*p[15+group*3+joint]
    return w


def initialize(filename,configuration,cases,height_penalty=10000.,objective='mean'):
    global POLICY,ENV,BASE,CASES,HEIGHT_PENALTY,OBJECTIVE
    POLICY=NumpyPolicy(filename);BASE=POLICY.weights['action_net.weight'].copy()
    ENV=QuadrupedEnv(Config(**configuration));CASES=cases
    HEIGHT_PENALTY=height_penalty;OBJECTIVE=objective


def candidate(parameters):
    POLICY.weights['action_net.weight']=parameter_weights(parameters,BASE)
    results=[]
    for options in CASES:
        ENV.path_name=options['path'];ENV.path=TrackPath(options['path'])
        obs,_=ENV.reset(seed=0,options=options);done=False;total=0.;begin=ENV.last_progress
        count=0;accel=rate=0.;previous=None;feet=np.zeros(4);air=np.zeros(4);zmin=1.;zmax=0.
        while not done:
            action=POLICY.predict(obs)[0];obs,r,d,t,info=ENV.step(action);done=d or t;total+=r
            if info['time_s']>=1.:
                count+=1;accel+=np.mean(np.square(info['joint_acceleration_rad_s2']))
                if previous is not None:rate+=np.mean(((np.array(info['torque_Nm'])-previous)/ENV.dt)**2)
                previous=np.array(info['torque_Nm']);feet=np.maximum(feet,info['feet_z'])
                air+=1-np.array(info['contacts']);zmin=min(zmin,info['position'][2]);zmax=max(zmax,info['position'][2])
        # Explicitly preserve visible stepping and forward motion during smoothing.
        progress=info['progress_m']-begin
        constraints=3000*sum(max(0,ENV.foot_radius+ENV.cfg.foot_clearance_min_m-v) for v in feet)+100*sum(max(0,.08-v/max(1,count)) for v in air)
        constraints+=200*max(0,min(max(0,ENV.path.total-begin-.1),.095*(ENV.cfg.seconds-ENV.cfg.clock_delay))-progress)
        constraints+=HEIGHT_PENALTY*max(0,zmax-zmin-.008)
        constraints+=100*float(np.sum(np.maximum(air/max(1,count)-.55,0)))
        results.append({'return':total-constraints,'raw_return':total,'steps':ENV.steps,'path':options['path'],
            'reason':info['reason'],'progress_m':progress,'joint_acceleration_rms_rad_s2':float(np.sqrt(accel/max(1,count))),
            'torque_rate_rms_Nm_s':float(np.sqrt(rate/max(1,count))), 'height_peak_to_peak_m':max(0,zmax-zmin),
            'feet_max_height_m':feet.tolist(),'air_fraction':(air/max(1,count)).tolist()})
    scores=[r['return'] for r in results]
    return {'return':float(min(scores) if OBJECTIVE=='worst' else np.mean(scores)),'steps':sum(r['steps'] for r in results),'cases':results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='runs/mini_lite_urdf/final/policy.zip');parser.add_argument('--output',default='runs/mini_lite_urdf/my_smoothing')
    parser.add_argument('--generations',type=int,default=32);parser.add_argument('--population',type=int,default=48)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--seconds',type=float,default=7.)
    parser.add_argument('--seed',type=int,default=81);parser.add_argument('--sigma',type=float,default=1.)
    parser.add_argument('--mean');parser.add_argument('--options')
    parser.add_argument('--height-penalty',type=float,default=10000.)
    parser.add_argument('--reward-version',type=int,choices=[3,4,5],default=5)
    parser.add_argument('--objective',choices=['mean','worst'],default='mean');args=parser.parse_args()
    if args.height_penalty<0:parser.error('--height-penalty must be nonnegative')
    model=NumpyPolicy(args.source);base=model.weights['action_net.weight'].copy()
    cfg=replace(Config(**model.pure_config),stage='path',seconds=args.seconds,reference_reward=False,reward_version=args.reward_version)
    cases=json.loads(Path(args.options).read_text(encoding='utf-8')) if args.options else [
        {'path':'straight','y_offset':.03}, {'path':'sine','start_progress':.6},
        {'path':'turn','start_progress':1.65}, {'path':'turn','start_progress':2.7}]
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory is not empty; choose a new --output to preserve earlier logs.')
    out.mkdir(parents=True,exist_ok=True)
    mean=np.zeros(33) if not args.mean else np.array(json.loads(Path(args.mean).read_text(encoding='utf-8')))
    std=np.r_[[.2,.12,.35],np.full(3,.06),np.full(3,.035),np.full(6,.09),np.tile([.08,.12,.12],2),
        np.tile([.04,.06,.06],2),[.04,.12,.12],[.04,.1,.1]]*args.sigma
    floor=std*.07;best=-np.inf;bestp=mean.copy();rng=np.random.default_rng(args.seed)
    steps=episodes=0
    (out/'config.json').write_text(json.dumps({'args':vars(args),'environment':asdict(cfg),'cases':cases,
        'training':'CEM scalar simulation rewards; neural full torques; no smoothing filter or PD'},indent=2),encoding='utf-8')
    worker_args=(args.source,asdict(cfg),cases,args.height_penalty,args.objective)
    if args.workers==1:initialize(*worker_args)
    pool=nullcontext(None) if args.workers==1 else ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=worker_args)
    with pool as executor:
        for generation in range(args.generations):
            parameters=rng.normal(mean,std,(args.population,len(mean)));parameters[0]=bestp
            if generation==0:parameters[1]=np.zeros(33)
            begin=time.perf_counter();rows=list(map(candidate,parameters) if executor is None else executor.map(candidate,parameters));scores=np.array([r['return'] for r in rows])
            steps+=sum(r['steps'] for r in rows);episodes+=args.population*len(cases);k=int(scores.argmax())
            if scores[k]>best:
                best=float(scores[k]);bestp=parameters[k].copy();model.weights['action_net.weight']=parameter_weights(bestp,base)
                provenance={'source':args.source,'generation':generation,'episodes':episodes,'environment_steps':steps,
                    'return':best,'parameters':bestp.tolist(),'metrics':rows[k]}
                model.save(out/'best_policy.zip',asdict(cfg),provenance)
                model.save(out/f'generation_{generation:03d}.zip',asdict(cfg),provenance)
                (out/'best_candidate.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
                (out/'best_parameters.json').write_text(json.dumps(bestp.tolist()),encoding='utf-8')
            with (out/'generations.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps({'generation':generation,'episodes':episodes,'steps':steps,'best_return':best,'candidates':rows})+'\n')
            print(json.dumps({'generation':generation,'best':round(best,3),'mean':round(float(scores.mean()),3),
                'acceleration':[round(r['joint_acceleration_rms_rad_s2'],1) for r in rows[k]['cases']],
                'height_mm':[round(r['height_peak_to_peak_m']*1000,2) for r in rows[k]['cases']],
                'seconds':round(time.perf_counter()-begin,1)}),flush=True)
            elite=parameters[np.argsort(scores)[-max(6,args.population//6):]]
            mean=.2*mean+.8*elite.mean(axis=0);std=np.maximum(.65*std+.35*elite.std(axis=0),floor)
            if (out/'STOP').exists():break
    (out/'search_status.json').write_text(json.dumps({'complete_episodes':episodes,'simulation_steps':steps,'best_return':best}),encoding='utf-8')


if __name__=='__main__':main()

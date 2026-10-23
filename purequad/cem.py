"""Reward-only search of neural torque weights; no control law or supervised labels."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .config import Config
from .model import assert_policy_compatible, LEGS, ASSETS
from zipfile import ZipFile
from .env import QuadrupedEnv
from .paths import Path as TrackPath

# Map physical search symmetries from the previous common -X HipX convention
# into the supplied URDF axes. Self joint-angle/velocity gains transform twice
# (input and torque), so their scalar coefficient stays unchanged.
def search_axis_signs():
    import xml.etree.ElementTree as ET
    root=ET.parse(ASSETS/'quadruped.xml').getroot()
    signs=[]
    for leg in LEGS:
        axis=np.fromstring(root.find(f".//joint[@name='{leg}_abduction']").get('axis'),sep=' ')
        dot=float(axis@np.array([-1.,0,0]))
        if abs(abs(dot)-1)>1e-12:raise ValueError('HipX axis needs a reviewed search mapping')
        signs.append(dot)
    return tuple(signs)

SEARCH_AXIS_SIGNS=search_axis_signs()

_ENV=None
_CORE=None


def worker_init(cfg,core):
    global _ENV,_CORE
    _ENV=QuadrupedEnv(Config(**cfg));_CORE=core


def worker(job):
    weight,bias,options=job
    if isinstance(options,list):
        cases=[worker((weight,bias,option)) for option in options]
        worst=min(cases,key=lambda c:c['return']);out=worst.copy();out['cases']=cases
        out['steps']=sum(c['steps'] for c in cases);return out
    env=_ENV
    if 'path' in options:env.path_name=options['path'];env.path=TrackPath(env.path_name)
    obs,_=env.reset(seed=0,options=options)
    total=0.;peak=0.;minheight=1.;bad=0;done=False;begin=env.last_progress
    feet=np.zeros(4);air=np.zeros(4);measured=0;zmin=1.;zmax=0.;cross_squared=0.;heading_squared=0.
    while not done:
        features=obs if weight.shape[1]==67 else np.r_[obs,np.outer(obs[[55,56]],obs[60:66]).ravel()]
        out=weight@features+bias
        if _CORE is not None:
            w1,b1,w2,b2,wa=_CORE
            latent=np.tanh(w2@np.tanh(w1@obs+b1)+b2);out+=wa@latent
        obs,r,d,t,info=env.step(np.clip(out,-1,1));total+=r;done=d or t
        cross_squared+=info['cross_track_m']**2;heading_squared+=info['heading_error_rad']**2
        peak=max(peak,max(abs(v) for v in info['rpy'][:2]));minheight=min(minheight,info['position'][2]);bad+=info['nonfoot_contacts']>0
        if info['time_s']>=1.:
            feet=np.maximum(feet,info['feet_z']);air+=1-np.array(info['contacts']);measured+=1
            zmin=min(zmin,info['position'][2]);zmax=max(zmax,info['position'][2])
    progress=info['progress_m']-begin;fractions=air/max(1,measured);constraints=0.
    if env.cfg.stage!='stand':
        constraints=3000*float(np.sum(np.maximum(env.foot_radius+env.cfg.foot_clearance_min_m-feet,0)))
        constraints+=100*float(np.sum(np.maximum(.08-fractions,0)+np.maximum(fractions-.55,0)))
        target_progress=min(max(0,env.path.total-begin-.1),max(0,.095*(env.cfg.seconds-env.cfg.clock_delay)))
        constraints+=200*max(0,target_progress-progress)
        if env.cfg.reward_version>=3:
            constraints+=10000*max(0,zmax-zmin-.008)+3000*max(0,peak-.08)
        if env.cfg.stage=='path':
            constraints+=2000*np.sqrt(cross_squared/env.steps)+200*np.sqrt(heading_squared/env.steps)
    return {'return':total-constraints,'raw_return':total,'constraint_penalty':constraints,'cross_track_rmse_m':float(np.sqrt(cross_squared/env.steps)),'heading_rmse_rad':float(np.sqrt(heading_squared/env.steps)),'height_peak_to_peak_m':max(0,zmax-zmin),'feet_max_height_m':feet.tolist(),'foot_air_fraction':fractions.tolist(),'reason':info['reason'],'duration_s':info['time_s'],'progress_m':info['progress_m']-begin,
        'max_tilt_rad':peak,'min_height_m':minheight,'nonfoot_contact_fraction':bad/env.steps,
        'joint_margin_rad':info['joint_margin_rad'],'steps':env.steps}


def parameter_weights(mode,parameters,base_w,base_b):
    w=base_w.copy();b=base_b.copy();p=np.asarray(parameters)
    for leg in range(4):
        side=1 if leg in (0,2) else -1
        for joint in range(3):
            row=leg*3+joint;direction=SEARCH_AXIS_SIGNS[leg] if joint==0 else 1
            sign=(side if joint==0 else 1)*direction
            if mode=='stand':
                b[row]=sign*p[joint]
                w[row,9+row]=p[3+joint];w[row,21+row]=p[6+joint];w[row,58]=sign*p[9+joint]
            elif mode=='motion':
                coeff=p[:18].reshape(3,6)[joint].copy()
                if leg in (1,2):coeff*=np.array([-1,-1,1,1,-1,-1])
                w[row,60:66]+=sign*coeff;b[row]+=sign*p[18+joint]
            elif mode=='feedback':
                w[row,9+row]+=p[joint];w[row,21+row]+=p[3+joint];b[row]+=sign*p[6+joint]
            elif mode=='steer':
                front=1 if leg in (0,1) else -1
                yaw_pattern=(front if joint==0 else side)*direction
                cross_pattern=(1 if joint==0 else side)*direction
                w[row,56]+=yaw_pattern*p[joint]
                w[row,55]+=cross_pattern*p[3+joint]
                w[row,8]+=yaw_pattern*p[6+joint]
                phase=np.array([-1,-1,1,1,-1,-1]) if leg in (1,2) else np.ones(6)
                w[row,73:79]+=yaw_pattern*phase*p[9:27].reshape(3,6)[joint]
                w[row,67:73]+=cross_pattern*phase*p[27:45].reshape(3,6)[joint]
            elif mode=='balance':
                front=1 if leg in (0,1) else -1
                roll_pattern=(1 if joint==0 else side)*direction
                pitch_pattern=front*direction
                vertical_pattern=(side if joint==0 else 1)*direction
                for group,(column,pattern) in enumerate([(1,roll_pattern),(0,pitch_pattern),(6,roll_pattern),(7,pitch_pattern),(58,vertical_pattern),(5,vertical_pattern)]):
                    w[row,column]+=pattern*p[group*3+joint]
            else:raise ValueError(mode)
    return w,b


def main():
    import torch
    from stable_baselines3 import PPO
    from .policy import TorquePolicy,PathTorquePolicy
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['stand','motion','feedback','steer','balance'],default='stand')
    parser.add_argument('--source');parser.add_argument('--output',default='runs/mini_lite_urdf/stand_search')
    parser.add_argument('--generations',type=int,default=40);parser.add_argument('--population',type=int,default=96)
    parser.add_argument('--workers',type=int,default=6);parser.add_argument('--seconds',type=float,default=4.)
    parser.add_argument('--seed',type=int,default=51);parser.add_argument('--sigma',type=float,default=1.)
    parser.add_argument('--options',default='{}');parser.add_argument('--options-file');parser.add_argument('--reference',action='store_true')
    parser.add_argument('--reward-version',type=int,default=1)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    cfg=Config(stage='stand' if args.mode=='stand' else 'motion',seconds=args.seconds,reference_reward=args.reference)
    cfg.reward_version=args.reward_version
    if args.source:
        with ZipFile(args.source) as archive:assert_policy_compatible(json.loads(archive.read('data'))['pure_config'])
        model=PPO.load(args.source,device='cpu');old=Config(**model.pure_config)
        cfg.period=old.period;cfg.action_scale=old.action_scale;cfg.clock_delay=old.clock_delay
        cfg.policy_variant=old.policy_variant
        if args.mode=='steer':
            cfg.stage='path';cfg.policy_variant='path'
            if model.policy.action_net.weight.shape[1]==131:
                fresh=PPO(PathTorquePolicy,QuadrupedEnv(cfg),policy_kwargs={'net_arch':[64,64]},device='cpu',verbose=0)
                weights=model.policy.state_dict();padded=torch.zeros_like(fresh.policy.action_net.weight)
                padded[:,:131]=weights['action_net.weight'];weights['action_net.weight']=padded
                fresh.policy.load_state_dict(weights);model=fresh
    else:
        env=QuadrupedEnv(cfg)
        model=PPO(TorquePolicy,env,n_steps=512,batch_size=256,n_epochs=10,
            policy_kwargs={'net_arch':[64,64]},gamma=.999,gae_lambda=.995,seed=args.seed,device='cpu',verbose=0)
        with torch.no_grad():model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_();model.policy.log_std.fill_(-3.5)
        env.close()
    fullw=model.policy.action_net.weight.detach().numpy().copy();base_w=fullw[:,64:].copy();base_b=model.policy.action_net.bias.detach().numpy().copy()
    core=None
    if np.any(fullw[:,:64]):
        net=model.policy.mlp_extractor.policy_net
        core=tuple(x.detach().numpy().copy() for x in [net[0].weight,net[0].bias,net[2].weight,net[2].bias])+ (fullw[:,:64],)
    if args.mode=='stand':
        mean=np.zeros(12);std=np.array([.12,.3,.25,3.,3.,3.,.6,1.,1.,.5,.5,.5]);floor=np.array([.003]*3+[.03]*3+[.02]*3+[.01]*3)
    elif args.mode=='motion':
        mean=np.zeros(21);std=np.r_[np.tile([.22,.22,.12,.12,.08,.08],3),[.04,.04,.04]];floor=np.r_[np.full(18,.006),np.full(3,.002)]
    elif args.mode=='feedback':
        mean=np.zeros(9);std=np.r_[np.full(3,.4),np.full(3,.2),np.full(3,.03)];floor=std*.05
    elif args.mode=='steer':
        mean=np.zeros(45);std=np.r_[[1.,1.,.7],[1.,1.5,1.],[.5,.6,.4],np.tile([.3,.3,.18,.18,.12,.12],6)];floor=std*.08
    else:
        mean=np.zeros(18);std=np.r_[np.tile([.3,.5,.5],2),np.tile([.1,.15,.15],2),[.15,.3,.3],[.2,.4,.4]];floor=std*.05
    std*=args.sigma;floor*=args.sigma
    best=-np.inf;bestp=mean.copy();episodes=steps=0;rng=np.random.default_rng(args.seed);options=json.loads(Path(args.options_file).read_text(encoding='utf-8')) if args.options_file else json.loads(args.options)
    (out/'config.json').write_text(json.dumps({'args':vars(args),'environment':asdict(cfg),
        'control':'Single neural network to 12 complete joint torques','training':'reward-only parameter search plus stepping/progress penalties; no teacher actions'},indent=2),encoding='utf-8')
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init,initargs=(asdict(cfg),core)) as executor:
        for generation in range(args.generations):
            params=rng.normal(mean,std,(args.population,len(mean)));params[0]=bestp
            jobs=[(*parameter_weights(args.mode,p,base_w,base_b),options) for p in params]
            begin=time.perf_counter();rows=list(executor.map(worker,jobs));scores=np.array([r['return'] for r in rows])
            episodes+=sum(len(r.get('cases',[r])) for r in rows);steps+=sum(r['steps'] for r in rows);k=int(np.argmax(scores))
            if scores[k]>best:
                best=float(scores[k]);bestp=params[k].copy();w,b=parameter_weights(args.mode,bestp,base_w,base_b)
                with torch.no_grad():model.policy.action_net.weight[:,64:].copy_(torch.tensor(w,dtype=torch.float32));model.policy.action_net.bias.copy_(torch.tensor(b,dtype=torch.float32))
                model.pure_config=asdict(cfg);model.reward_search={'mode':args.mode,'source':args.source,'generation':generation,
                    'episodes':episodes,'simulation_steps':steps,'parameters':bestp.tolist(),'return':best,'candidate_metrics':rows[k]}
                model.save(out/'best_policy');(out/'best_candidate.json').write_text(json.dumps(model.reward_search,indent=2),encoding='utf-8')
            with (out/'generations.jsonl').open('a',encoding='utf-8') as h:h.write(json.dumps({'generation':generation,'episodes':episodes,'steps':steps,'best_return':best,'candidates':rows})+'\n')
            print(f"gen={generation} best={best:.2f} mean={scores.mean():.2f} duration={rows[k]['duration_s']:.2f}s progress={rows[k]['progress_m']:.3f}m reason={rows[k]['reason']} elapsed={time.perf_counter()-begin:.1f}s",flush=True)
            elite=params[np.argsort(scores)[-max(8,args.population//6):]]
            mean=.2*mean+.8*elite.mean(axis=0);std=np.maximum(.6*std+.4*elite.std(axis=0),floor)
            if (out/'STOP').exists():break
    (out/'search_status.json').write_text(json.dumps({'complete_episodes':episodes,'simulation_steps':steps,'best_return':best},indent=2),encoding='utf-8')


if __name__=='__main__':main()

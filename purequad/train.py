"""PPO trained with rewards only; the reference motion never supplies action labels."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from .env import QuadrupedEnv
from .config import Config
from .model import assert_policy_compatible
from zipfile import ZipFile
from .paths import Path as TrackPath
from .policy import TorquePolicy,PathTorquePolicy
from .evaluate import rollout


class MixedEnv(QuadrupedEnv):
    def reset(self,*,seed=None,options=None):
        if seed is not None:self.np_random=np.random.default_rng(seed)
        if self.cfg.stage=='path':
            self.path_name=str(self.np_random.choice(['straight','sine','turn']));self.path=TrackPath(self.path_name)
            if options is None:options={'start_progress':float(self.np_random.uniform(0,self.path.total-.6))}
        return super().reset(seed=seed,options=options)


class Validation(BaseCallback):
    def __init__(self,cfg,out,freq,seconds):
        super().__init__();self.cfg=cfg;self.out=out;self.freq=freq;self.seconds=seconds;self.next=freq;self.best=-np.inf
    def _evaluate(self):
        e=QuadrupedEnv(replace(self.cfg,seconds=self.seconds,reference_reward=False))
        try:r,_=rollout(self.model,e)
        finally:e.close()
        r['num_timesteps']=self.num_timesteps
        penalty=0.
        if self.cfg.stage!='stand':
            penalty=3000*sum(max(0,e.foot_radius+self.cfg.foot_clearance_min_m-h) for h in r['feet_max_height_m'])
            penalty+=100*sum(max(0,.08-v)+max(0,v-.55) for v in r['foot_air_fraction'])
            penalty+=200*max(0,.095*(self.seconds-self.cfg.clock_delay)-r['progress_m'])
        score=r['return']-penalty;r['selection_score']=score;r['constraint_penalty']=penalty
        with (self.out/'evaluations.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
        if score>self.best:
            self.best=score;self.model.save(self.out/'best_policy');(self.out/'best_evaluation.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
        self.model.save(self.out/f'checkpoint_{self.num_timesteps}')
        print(f"EVAL {self.num_timesteps} score={score:.2f} duration={r['duration_s']:.1f}s progress={r['progress_m']:.3f}m height={r['min_height_m']:.3f} tilt={r['max_tilt_rad']:.3f} feet={np.round(r['feet_max_height_m'],3)} reason={r['reason']}",flush=True)

    def _on_training_start(self):
        self._evaluate()

    def _on_step(self):
        if self.n_calls%128==0 and (self.out/'STOP').exists():return False
        if self.num_timesteps<self.next:return True
        self.next=self.num_timesteps+self.freq
        self._evaluate()
        return True



def main():
    p=argparse.ArgumentParser();p.add_argument('--source');p.add_argument('--stage',choices=['stand','motion','path'],default='motion')
    p.add_argument('--steps',type=int,default=1000000);p.add_argument('--envs',type=int,default=4);p.add_argument('--seconds',type=float,default=12)
    p.add_argument('--output',default='runs/mini_lite_urdf/motion_ppo');p.add_argument('--reference',action='store_true')
    p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--log-std',type=float,default=-3.5)
    p.add_argument('--n-steps',type=int,default=1024);p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--eval-freq',type=int,default=50000);p.add_argument('--eval-seconds',type=float,default=12)
    p.add_argument('--seed',type=int,default=61);args=p.parse_args();torch.set_num_threads(1)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.source:
        with ZipFile(args.source) as archive:assert_policy_compatible(json.loads(archive.read('data'))['pure_config'])
    source=PPO.load(args.source,device='cpu') if args.source else None
    cfg=replace(Config(**source.pure_config),stage=args.stage,seconds=args.seconds,reference_reward=args.reference) if source else Config(stage=args.stage,seconds=args.seconds,reference_reward=args.reference)
    cfg.reset_noise=.002
    cfg.reward_version=max(2,cfg.reward_version)
    env=DummyVecEnv([lambda i=i:Monitor(MixedEnv(cfg),str(out/f'env_{i}.monitor.csv'),info_keywords=('reason','is_success')) for i in range(args.envs)]);env.seed(args.seed)
    policy_class=PathTorquePolicy if cfg.policy_variant=='path' else TorquePolicy
    model=PPO(policy_class,env,policy_kwargs={'net_arch':[64,64]},n_steps=args.n_steps,batch_size=256,n_epochs=args.epochs,
        learning_rate=args.learning_rate,gamma=.9995,gae_lambda=.995,ent_coef=0.,target_kl=.02,seed=args.seed,device='cpu',verbose=0)
    if source:model.policy.load_state_dict(source.policy.state_dict())
    with torch.no_grad():model.policy.log_std.fill_(args.log_std);model.policy.value_net.weight.zero_();model.policy.value_net.bias.zero_()
    model.pure_config=asdict(cfg);model.training_source=args.source;model.set_logger(configure(str(out),['csv']))
    (out/'config.json').write_text(json.dumps({'args':vars(args),'environment':asdict(cfg),'training':'PPO rewards only; 12 full joint torques'},indent=2),encoding='utf-8')
    try:model.learn(args.steps,callback=Validation(cfg,out,args.eval_freq,args.eval_seconds))
    finally:
        model.save(out/'policy');env.close();(out/'training_status.json').write_text(json.dumps({'num_timesteps':model.num_timesteps,'complete':not (out/'STOP').exists()},indent=2),encoding='utf-8')


if __name__=='__main__':main()

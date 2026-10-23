import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from stable_baselines3 import PPO
from purequad.numpy_policy import NumpyPolicy
import numpy as np


if __name__=='__main__':
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('--model',default='runs/mini_lite_urdf/final/policy.zip');parser.add_argument('--output',default='results/mini_lite_urdf/checkpoint_compatibility.json');args=parser.parse_args();filename=args.model
    numpy=NumpyPolicy(filename);pytorch=PPO.load(filename,device='cpu')
    maximum=0.
    for seed in range(32):
        obs=np.random.default_rng(seed).normal(0,.2,67).astype(np.float32)
        a=pytorch.predict(obs,deterministic=True)[0];b=numpy.predict(obs)[0]
        maximum=max(maximum,float(np.max(np.abs(a-b))))
    assert maximum<2e-6,maximum
    result={'SB3_checkpoint_load':'PASS','observations':67,'actions':12,'test_observations':32,
        'max_normalized_action_difference':maximum,'reward_version':pytorch.pure_config['reward_version']}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

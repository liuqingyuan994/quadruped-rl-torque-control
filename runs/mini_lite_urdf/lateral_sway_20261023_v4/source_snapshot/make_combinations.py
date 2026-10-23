"""Use central finite-difference responses to propose bounded policy experiments.

This model proposes candidates only. All selections use actual simulation results.
"""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent
cfg=json.loads((ROOT/'01_probe/config.json').read_text())
rows=[json.loads(s) for s in (ROOT/'01_probe/candidates.jsonl').read_text().splitlines()]
assert len(rows)==len(cfg['parameters'])
keys=list(cfg['objective'])
p=np.asarray([r['parameters'] for r in rows]);rat=np.asarray([[r['metrics']['ratios'][k] for k in keys] for r in rows])
axes=np.where(np.any(p!=0,axis=0))[0];scale=np.max(abs(p),axis=0)[axes]
gradient=[];curvature=[]
for ax in axes:
    low=np.where(p[:,ax]<0)[0][0];high=np.where(p[:,ax]>0)[0][0]
    gradient.append((rat[high]-rat[low])/2)
    curvature.append((rat[high]+rat[low])/2-1)
gradient=np.asarray(gradient).T;curvature=np.asarray(curvature).T
# Remove axes with no effect and large nonlinear responses (e.g. fall on one side).
valid=(np.max(abs(gradient),axis=0)>1e-5)&(np.max(abs(curvature),axis=0)<.4)
axes=axes[valid];scale=scale[valid];gradient=gradient[:,valid];curvature=curvature[:,valid]
objective=np.asarray([cfg['objective'][k] for k in keys])
limits={'roll_cycle_rad':1.00,'pitch_cycle_rad':1.005,'height_cycle_m':1.005,
    'roll_std_rad':1.03,'pitch_std_rad':1.015,'height_std_m':1.015,'HF10_rad':1.005,
    'yaw_gait_rms_rad':.99,'lateral_gait_rms_m':.99,
    'acceleration_rms_rad_s2':1.015,'torque_rate_rms_Nm_s':1.015}
ix=[keys.index(k) for k in limits];lim=np.asarray(list(limits.values()))
def predict(x):return 1+gradient@x+curvature@(x*x)
rng=np.random.default_rng(10234)
candidates=[np.zeros(86)]
feasible=sorted((r for r in rows if r['metrics']['feasible']),key=lambda r:r['metrics']['score'],reverse=True)
for row in feasible[:6]:candidates.append(np.asarray(row['parameters']))
for bound in [1.,1.5,2.,2.5]:
    weight=objective[None,:]*rng.uniform(.7,1.4,(8,len(keys)))
    penalty=rng.choice([.005,.015,.03],8)[:,None]
    x=np.zeros((8,len(axes)));m=x.copy();v2=x.copy()
    for step in range(1,2501):
        pred=1+x@gradient.T+(x*x)@curvature.T
        d=2*weight*pred
        d[:,ix]+=2000*np.maximum(0,pred[:,ix]-lim)
        derivative=d@gradient+2*x*(d@curvature)+2*penalty*x
        m=.9*m+.1*derivative;v2=.999*v2+.001*derivative**2
        x=np.clip(x-.025*(m/(1-.9**step))/(np.sqrt(v2/(1-.999**step))+1e-8),-bound,bound)
    for result in x:
        v=np.zeros(86);v[axes]=result*scale
        v=np.clip(v,-.5,.5)
        if not any(np.linalg.norm(v-c)<.02 for c in candidates):candidates.append(v)
    print('bound',bound,'candidates',len(candidates))
for row in feasible[:6]:
    v=np.asarray(row['parameters'])
    for n in [2,3,4]:candidates.append(np.clip(v*n,-.5,.5))
(ROOT/'combination_parameters.json').write_text(json.dumps([v.tolist() for v in candidates]),encoding='utf-8')
print('Candidate count',len(candidates),'axes',len(axes))

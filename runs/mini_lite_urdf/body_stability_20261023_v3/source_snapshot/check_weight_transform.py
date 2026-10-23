"""Check zero-edit identity and exact raw-actor retention coefficients."""
from pathlib import Path
import json
import numpy as np
import body_cycle_core as search

root=Path(__file__).resolve().parent
policy=search.NumpyPolicy(root/'baseline_policy.zip')
base=policy.weights['action_net.weight'];bias=policy.weights['action_net.bias']
np.testing.assert_array_equal(search.weights(base,np.zeros(65),bias),base)
rng=np.random.default_rng(2310);maximum=0.
for _ in range(128):
    p=rng.uniform(-.03,.03,65);p[62:]=rng.uniform(0,.5,3)
    raw_p=p.copy();raw_p[62:]=0
    raw_w=search.weights(base,raw_p,bias);mixed_w=search.weights(base,p,bias)
    obs=rng.normal(0,.1,67).astype(np.float32);obs[66]=1
    hidden=rng.normal(0,.1,64).astype(np.float32)
    features=np.r_[hidden,obs,np.outer(obs[[55,56]],obs[60:66]).ravel()].astype(np.float32)
    alpha=np.tile(p[62:],4)
    expected=(1-alpha)*(raw_w@features+bias)+alpha*obs[33:45]/.4
    actual=mixed_w@features+bias
    maximum=max(maximum,float(np.max(np.abs(actual-expected))))
assert maximum<2e-6,maximum
result={'status':'PASS','zero_edit_bitwise_identity':True,'trials':128,
    'max_raw_action_difference':maximum,'existing_previous_action_input_scale':.4,
    'runtime_architecture_changed':False}
(root/'weight_transform_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))

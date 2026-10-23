"""Verify zero-edit identity and exact replay of the selected weight edit."""
from pathlib import Path
import json,os
os.environ['MINILITE_PROJECT']='D:/Download/桌面/rl'
import numpy as np
import lateral_sway_core as core
ROOT=Path(__file__).resolve().parent
a=core.NumpyPolicy(ROOT/'baseline_policy.zip');b=core.NumpyPolicy(ROOT/'selected_policy.zip')
w=a.weights['action_net.weight'];bias=a.weights['action_net.bias']
np.testing.assert_array_equal(core.weights(w,np.zeros(86),bias),w)
params=json.loads((ROOT/'selected_metadata.json').read_text(encoding='utf-8'))['parameters']
expected=core.weights(w,params,bias)
np.testing.assert_array_equal(expected,b.weights['action_net.weight'])
assert a.pure_config==b.pure_config
changed=[k for k in a.weights if not np.array_equal(a.weights[k],b.weights[k])]
assert changed==['action_net.weight']
assert np.isfinite(expected).all() and len(core.PARAMETER_NAMES)==86
result={'status':'PASS','zero_edit_identity':True,'selected_parameter_replay':'BITWISE_EQUAL',
    'changed_tensors':changed,'configuration_changed':False,'parameters':86}
(ROOT/'weight_transform_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(result)

"""Local fine search around the first full nine-case passing candidate."""
from pathlib import Path
import json,itertools
ROOT=Path(__file__).resolve().parent
base=json.loads((ROOT/'trial_03_22.json').read_text())['parameters']
params=[[0.]*86,base]
for yawhip,heading,phase in itertools.product([.5,.65,.8,1.0],[0.,.25],[-.025,-.045]):
    v=base.copy();extra=yawhip-.5
    v[26]=extra;v[65]=max(-.5,-.25-extra);v[66]=.5;v[67]=-extra
    v[83]=phase;v[27]=heading;params.append(v)
(ROOT/'refinement_parameters.json').write_text(json.dumps(params),encoding='utf-8')
(ROOT/'full_screen_cases.json').write_text(json.dumps([{'path':name,'seconds':60} for name in ['straight','sine','turn']]+[{'path':'straight','y_offset':-.05,'seconds':60}]),encoding='utf-8')
print('refinement candidates',len(params))

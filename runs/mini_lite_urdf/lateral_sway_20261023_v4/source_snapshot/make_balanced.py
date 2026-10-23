from pathlib import Path
import json,itertools
ROOT=Path(__file__).resolve().parent
base=json.loads((ROOT/'trial_03_22.json').read_text())['parameters']
params=[[0.]*86,base]
for heading,yawabd,abdharm,hipbalance in itertools.product([.08,.15],[-.25,-.5],[-.06,-.12],[0.,.015]):
    v=base.copy();v[27]=heading;v[65]=yawabd;v[81]=abdharm;v[21]=hipbalance;v[8]=.02
    params.append(v)
(ROOT/'balanced_parameters.json').write_text(json.dumps(params),encoding='utf-8')
print('balanced candidates',len(params))

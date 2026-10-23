from pathlib import Path
import json,itertools
ROOT=Path(__file__).resolve().parent
base=json.loads((ROOT/'trial_03_22.json').read_text())['parameters']
params=[[0.]*86,base]
for heading,kneeamp in itertools.product([.10,.125,.15,.175,.20],[.012,.02]):
    v=base.copy();v[27]=heading;v[8]=kneeamp;params.append(v)
for cross,abd in itertools.product([-.15,.15,.3],[-.02,0.]):
    v=base.copy();v[27]=.15;v[28]=cross;v[81]=abd;params.append(v)
(ROOT/'heading_balance_parameters.json').write_text(json.dumps(params),encoding='utf-8')
print('heading-balance candidates',len(params))

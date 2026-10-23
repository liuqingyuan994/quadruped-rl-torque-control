from pathlib import Path
import json,itertools
ROOT=Path(__file__).resolve().parent
params=[[0.]*86]
for yawhip,phase,hipamp,hip2 in itertools.product([.25,.5],[-.025,-.05],[.015,.03,.045],[0.,.03]):
    row=[0.]*86;row[4]=.03;row[22]=-.025;row[65]=-.25;row[66]=yawhip;row[83]=phase;row[7]=hipamp;row[10]=hip2
    params.append(row)
(ROOT/'clearance_parameters.json').write_text(json.dumps(params),encoding='utf-8')
print('clearance-preserving candidate count',len(params))

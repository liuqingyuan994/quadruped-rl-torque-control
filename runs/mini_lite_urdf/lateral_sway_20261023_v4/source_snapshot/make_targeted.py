from pathlib import Path
import json,itertools
ROOT=Path(__file__).resolve().parent
params=[[0.]*86]
for yawhip,phase,knee,hipclock in itertools.product([.25,.5],[-.025,-.05],[0.,-.025],[0.,-.05]):
    row=[0.]*86;row[65]=-.25;row[66]=yawhip;row[83]=phase;row[22]=knee;row[10]=hipclock;row[4]=.03
    params.append(row)
    v=row.copy();v[8]=.012;params.append(v)
for yawabd in [-.25,-.5]:
    for phase in [-.025,-.05,-.075]:
        for hipclock in [0.,-.05]:
            row=[0.]*86;row[65]=yawabd;row[66]=.25;row[83]=phase;row[10]=hipclock
            params.append(row)
(ROOT/'targeted_parameters.json').write_text(json.dumps(params),encoding='utf-8')
print('targeted candidates',len(params))

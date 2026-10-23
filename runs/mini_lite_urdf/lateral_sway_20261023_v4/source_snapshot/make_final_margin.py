from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
base=json.loads((ROOT/'heading_balance_parameters.json').read_text())[4]
params=[[0.]*86,base]
for axis,values in {3:[.02,.04,.06],5:[.015,.03,.045],7:[.005,.015,.025],8:[.008,.016,.025],10:[.01,.02,.03],21:[-.005,-.01,.005],22:[-.005,-.01,.005],65:[-.1,-.2],81:[-.01,-.02,.01]}.items():
    for value in values:
        row=base.copy();row[axis]+=value;params.append(row)
for q_a,q_k in [(.02,.015),(.04,.03),(.04,.015),(.02,.03)]:
    row=base.copy();row[3]+=q_a;row[5]+=q_k;params.append(row)
(ROOT/'margin_parameters.json').write_text(json.dumps(params),encoding='utf-8')
(ROOT/'margin_cases.json').write_text(json.dumps([{'path':'straight','y_offset':-.05,'seconds':60}]),encoding='utf-8')
print('margin candidates',len(params))

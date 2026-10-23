from pathlib import Path
import json,sys,os,hashlib
os.environ['MINILITE_PROJECT']='D:/Download/桌面/rl'
import lateral_sway_core as core
ROOT=Path(__file__).resolve().parent
folder,index,dest=sys.argv[1],int(sys.argv[2]),sys.argv[3]
row=[json.loads(s) for s in (ROOT/folder/'candidates.jsonl').read_text().splitlines()][index]
assert row['index']==index
source=ROOT/'baseline_policy.zip';p=core.NumpyPolicy(source)
p.weights['action_net.weight']=core.weights(p.weights['action_net.weight'],row['parameters'],p.weights['action_net.bias'])
meta={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'method':'lateral_yaw_roll_constrained_neural_search','search_folder':folder,
      'candidate_index':index,'parameters':row['parameters'],'metrics':row['metrics']}
p.save(ROOT/(dest+'.zip'),p.pure_config,meta)
(ROOT/(dest+'.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print('exported',dest)

"""Measured combinations of the individually beneficial neural-weight directions."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import time
import numpy as np
import body_stability_train as search


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();root=Path(args.root);out=root/'02_directed'
    out.mkdir();source=root/'baseline_policy.zip'
    configuration=json.loads((root/'01_screen/config.json').read_text(encoding='utf-8'))
    reference=json.loads((root/'01_screen/reference.json').read_text(encoding='utf-8'))
    parameters=[np.zeros(32)]
    for damping,knee_position,knee_third in itertools.product([.15,.30,.5],[0,.03,.06],[0,-.12,-.24]):
        a=np.zeros(32);a[0]=damping*.3;a[5]=knee_position;a[11]=-damping*.1
        a[12]=-damping*.25;a[14]=knee_third;a[17:20]=damping
        parameters.append(a)
    parameters=np.asarray(parameters)
    search.save_json(out/'config.json',{'source':str(source),'configuration':configuration['environment'],
        'cases':configuration['cases'],'reference':reference,'parameters':parameters.tolist()})
    started=time.perf_counter()
    with ProcessPoolExecutor(args.workers,initializer=search.initialize,
            initargs=(str(source),configuration['environment'],configuration['cases'],reference)) as pool:
        rows=[]
        for i,row in enumerate(pool.map(search.candidate,parameters)):
            rows.append(row)
            with (out/'candidates.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps({'index':i,'parameters':parameters[i].tolist(),'metrics':row})+'\n')
            if (i+1)%4==0: print(json.dumps({'completed':i+1,'total':len(parameters),'best':max(r['score'] for r in rows),'elapsed':time.perf_counter()-started}),flush=True)
    model=search.NumpyPolicy(source);base=model.weights['action_net.weight'].copy()
    ranks=np.argsort([r['score'] for r in rows])[::-1]
    for rank,index in enumerate(ranks[:8]):
        model.weights['action_net.weight']=search.weights(base,parameters[index])
        meta={'source':str(source),'method':'measured_body_weight_direction_combinations','parameters':parameters[index].tolist(),'metrics':rows[index]}
        model.save(out/f'rank_{rank}.zip',model.pure_config,meta)
        search.save_json(out/f'rank_{rank}.json',meta)
    search.save_json(out/'best_parameters.json',parameters[ranks[0]].tolist())
    print(json.dumps({'best':rows[ranks[0]],'elapsed':time.perf_counter()-started}),flush=True)


if __name__=='__main__':main()

"""Repeatable constrained search on an existing direct-torque neural policy.

Inputs/outputs are explicit; runtime physics and acceptance limits are unchanged.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

PROJECT = Path(os.environ.get('MINILITE_PROJECT', str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(PROJECT / 'tools'))
import body_cycle_core as search


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True); p.add_argument('--output', required=True)
    p.add_argument('--parameters'); p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seconds', type=float, default=12); p.add_argument('--full-paths', action='store_true')
    p.add_argument('--offsets', action='store_true', help='Also screen straight paths with +/- 5 cm initial offset')
    p.add_argument('--cases', help='JSON file with explicit reset options and episode lengths')
    args = p.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    source = Path(args.source).resolve(); policy = search.NumpyPolicy(source)
    cfg = asdict(replace(search.Config(**policy.pure_config), stage='path', seconds=args.seconds, reference_reward=False))
    cases = ([{'path': name, 'seconds': 60} for name in ['straight', 'sine', 'turn']] if args.full_paths else
        [{'path': 'straight', 'seconds': args.seconds}, {'path': 'sine', 'start_progress': .6, 'seconds': args.seconds},
         {'path': 'turn', 'start_progress': 1.65, 'seconds': args.seconds}, {'path': 'turn', 'start_progress': 2.7, 'seconds': args.seconds}])
    if args.offsets:
        cases += [{'path': 'straight', 'y_offset': offset, 'seconds': args.seconds} for offset in [-.05, .05]]
    if args.cases:
        cases = json.loads(Path(args.cases).read_text(encoding='utf-8'))
    if args.parameters:
        parameters = np.array(json.loads(Path(args.parameters).read_text(encoding='utf-8')))
    else:
        parameters = [np.zeros(62)]
        axes = {0:.08,1:.06,2:.08,4:.04,5:.04,8:.03,11:.08,14:.12,17:.25,18:.25,19:.25,20:.12,26:.15}
        axes.update({i:.02 for i in range(32,41)})
        axes.update({i:.04 for i in range(41,47)})
        axes.update({i:.025 for i in range(47,53)})
        axes.update({i:.25 for i in range(53,59)})
        axes.update({i:.2 for i in range(59,62)})
        for axis,delta in axes.items():
            for sign in [-1,1]:
                row=np.zeros(62);row[axis]=sign*delta;parameters.append(row)
        parameters=np.asarray(parameters)
    if parameters.ndim==2 and parameters.shape[1]==62:parameters=np.pad(parameters,((0,0),(0,3)))
    if parameters.ndim != 2 or parameters.shape[1] != 65 or not np.isfinite(parameters).all():
        raise ValueError('Expected finite N x 65 parameters')
    if np.max(abs(parameters)) > .5: raise ValueError('Parameter clipping would alter the requested experiment')
    started = time.perf_counter()
    search.save_json(out/'config.json', {'args': vars(args), 'environment': cfg, 'cases': cases,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'model_sha256': hashlib.sha256((PROJECT/'purequad/assets/quadruped.xml').read_bytes()).hexdigest(),
        'parameter_names': search.PARAMETER_NAMES, 'parameters': parameters.tolist(), 'objective': search.OBJECTIVE,
        'core_source_sha256':hashlib.sha256(Path(search.__file__).read_bytes()).hexdigest()})
    with ProcessPoolExecutor(args.workers, initializer=search.initialize, initargs=(str(source), cfg, cases)) as pool:
        reference = list(pool.map(search.measure_case, cases))
    search.save_json(out/'reference.json', reference)
    print(json.dumps({'event': 'baseline_ready', 'count': len(parameters), 'score': search.score(reference, reference)['score']}), flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=search.initialize, initargs=(str(source), cfg, cases, reference)) as pool:
        for i, result in enumerate(pool.map(search.candidate, parameters)):
            rows.append(result)
            record = {'index': i, 'parameters': parameters[i].tolist(), 'metrics': result}
            with (out/'candidates.jsonl').open('a', encoding='utf-8') as log: log.write(json.dumps(record)+'\n')
            state = {'active': True, 'completed': i+1, 'total': len(parameters),
                'episodes': len(cases)*(i+1), 'steps': sum(row['steps'] for row in rows),
                'best_score': max(row['score'] for row in rows), 'elapsed_s': time.perf_counter()-started}
            search.save_json(out/'training_state.json', state)
            if (i+1)%6 == 0: print(json.dumps(state), flush=True)
    base = policy.weights['action_net.weight'].copy()
    ranks = np.argsort([r['score'] for r in rows])[::-1]
    for rank, index in enumerate(ranks[:10]):
        policy.weights['action_net.weight'] = search.weights(base, parameters[index], policy.weights['action_net.bias'])
        metadata = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'method': 'body_cycle_and_joint_vibration_constrained_neural_search', 'parameters': parameters[index].tolist(),
            'metrics': rows[index], 'candidate_index': int(index)}
        policy.save(out/f'rank_{rank}.zip', policy.pure_config, metadata)
        search.save_json(out/f'rank_{rank}.json', metadata)
    state['active'] = False; search.save_json(out/'training_state.json', state)
    print(json.dumps({'event': 'complete', 'best_index': int(ranks[0]), 'score': rows[ranks[0]]['score'],
        'ratios': rows[ranks[0]]['ratios'], 'elapsed_s': time.perf_counter()-started}), flush=True)


if __name__ == '__main__': main()

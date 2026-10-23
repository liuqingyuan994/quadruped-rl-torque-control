"""Refine frozen direct-torque policy weights using measured body vibration.

The runtime environment, timing, actuator scaling and physical model are read-only.
All candidate changes are baked into the existing actor's output weight matrix.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace, asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

PROJECT = Path(os.environ.get('MINILITE_PROJECT', 'D:/Download/桌面/rl'))
sys.path.insert(0, str(PROJECT))
from purequad.config import Config
from purequad.env import QuadrupedEnv
from purequad.numpy_policy import NumpyPolicy
from purequad.paths import Path as Track
from purequad.metrics import high_frequency_rms

POLICY = ENV = BASE = CASES = REFERENCE = None
PARAMETER_NAMES = [
    'joint_dq_abduction', 'joint_dq_hip', 'joint_dq_knee',
    'joint_q_abduction', 'joint_q_hip', 'joint_q_knee',
    'clock1_abduction', 'clock1_hip', 'clock1_knee',
    'clock2_abduction', 'clock2_hip', 'clock2_knee',
    'clock3_abduction', 'clock3_hip', 'clock3_knee',
    'roll_state', 'pitch_state', 'roll_rate', 'pitch_rate',
    'vertical_velocity', 'height_error', 'front_rear_hip_amplitude',
    'front_rear_knee_amplitude', 'front_rear_hip_phase',
    'front_rear_knee_phase', 'hip_knee_relative_phase',
    'yaw_rate', 'heading_error', 'cross_track_error',
    'hidden_abduction', 'hidden_hip', 'hidden_knee']


def weights(base, parameters):
    p = np.clip(np.asarray(parameters, dtype=float), -.5, .5)
    if p.shape != (32,):
        raise ValueError('Expected 32 local neural weight parameters')
    w = base.copy()
    for leg in range(4):
        front = 1 if leg < 2 else -1
        for joint in range(3):
            row = 3 * leg + joint
            w[row, 85 + row] *= np.exp(p[joint])
            w[row, 73 + row] *= np.exp(p[3 + joint])
            w[row, :64] *= np.exp(p[29 + joint])
            for harmonic in range(3):
                column = 124 + 2 * harmonic
                w[row, column:column + 2] *= np.exp(p[6 + 3 * harmonic + joint])
                theta = 0.
                if joint == 1:
                    theta = front * p[23] + p[25]
                elif joint == 2:
                    theta = front * p[24] - p[25]
                theta *= harmonic + 1
                a, b = w[row, column:column + 2].copy()
                w[row, column] = a * np.cos(theta) - b * np.sin(theta)
                w[row, column + 1] = a * np.sin(theta) + b * np.cos(theta)
            if joint in (1, 2):
                w[row, 124:130] *= np.exp(front * p[20 + joint])
    for index, column in enumerate([65, 64, 70, 71, 69, 122], 15):
        w[:, column] *= np.exp(p[index])
    w[:, 72] *= np.exp(p[26])
    w[:, 120] *= np.exp(p[27]); w[:, 137:143] *= np.exp(p[27])
    w[:, 119] *= np.exp(p[28]); w[:, 131:137] *= np.exp(p[28])
    return w


def initialize(source, configuration, cases, reference=None):
    global POLICY, ENV, BASE, CASES, REFERENCE
    POLICY = NumpyPolicy(source)
    BASE = POLICY.weights['action_net.weight'].copy()
    assert BASE.shape == (12, 143)
    np.testing.assert_array_equal(weights(BASE, np.zeros(32)), BASE)
    ENV = QuadrupedEnv(Config(**configuration))
    CASES, REFERENCE = cases, reference


def measure_case(options):
    ENV.path_name = options['path']; ENV.path = Track(options['path'])
    ENV.cfg.seconds = float(options.get('seconds', 12))
    obs, _ = ENV.reset(seed=17, options=options)
    begin = ENV.last_progress; done = False; total = 0.
    body = []; joints = []; torques = []; accelerations = []
    feet = np.zeros(4); air = np.zeros(4); count = 0
    zmin, zmax, min_height, max_tilt, max_speed = 1., 0., 1., 0., 0.
    cross_square = cross_max = bad = 0.
    while not done:
        obs, reward, terminated, truncated, info = ENV.step(POLICY.predict(obs)[0])
        done = terminated or truncated; total += reward
        cross_square += info['cross_track_m'] ** 2
        cross_max = max(cross_max, abs(info['cross_track_m']))
        min_height = min(min_height, info['position'][2])
        max_tilt = max(max_tilt, max(abs(x) for x in info['rpy'][:2]))
        max_speed = max(max_speed, max(abs(x) for x in info['joint_velocity_rad_s']))
        bad += info['nonfoot_contacts'] > 0
        if info['time_s'] >= 1:
            count += 1; feet = np.maximum(feet, info['feet_z'])
            air += 1 - np.asarray(info['contacts'])
            zmin = min(zmin, info['position'][2]); zmax = max(zmax, info['position'][2])
        if info['time_s'] >= 3:
            body.append([*info['rpy'][:2], info['position'][2], info['heading_error_rad'], info['cross_track_m']])
            joints.append(info['joint_position_rad']); torques.append(info['torque_Nm'])
            accelerations.append(info['joint_acceleration_rad_s2'])
    result = {'path': options['path'], 'options': options, 'reason': info['reason'],
        'duration_s': info['time_s'], 'progress_m': info['progress_m'] - begin,
        'steps': ENV.steps, 'raw_return': total, 'cross_track_rmse_m': float(np.sqrt(cross_square / ENV.steps)),
        'cross_track_max_m': cross_max, 'min_height_m': min_height, 'max_tilt_rad': max_tilt,
        'max_joint_speed_rad_s': max_speed, 'nonfoot_contact_fraction': bad / ENV.steps,
        'height_peak_to_peak_m': max(0., zmax - zmin), 'feet_max_height_m': feet.tolist(),
        'air_fraction': (air / max(1, count)).tolist()}
    names = ['roll_std_rad', 'pitch_std_rad', 'height_std_m', 'yaw_gait_rms_rad', 'lateral_gait_rms_m',
             'roll_rate_rms_rad_s', 'pitch_rate_rms_rad_s', 'vertical_speed_rms_m_s',
             'roll_HF8_rad', 'pitch_HF8_rad', 'height_HF8_m', 'HF10_rad',
             'acceleration_rms_rad_s2', 'torque_rate_rms_Nm_s']
    if len(body) < 8:
        result.update({name: 1e3 for name in names})
        return result
    b = np.asarray(body); q = np.asarray(joints); tau = np.asarray(torques)
    std = np.std(b[:, :3], axis=0)
    rates = np.sqrt(np.mean((np.diff(b[:, :3], axis=0) / ENV.dt) ** 2, axis=0))
    high = high_frequency_rms(b[:, :3], ENV.dt, 8)
    result.update(dict(zip(names[:3], map(float, std))))
    result.update(yaw_gait_rms_rad=float(high_frequency_rms(b[:, 3:4], ENV.dt, .7)[0]),
        lateral_gait_rms_m=float(high_frequency_rms(b[:, 4:5], ENV.dt, .7)[0]))
    result.update(dict(zip(names[5:8], map(float, rates))))
    result.update(dict(zip(names[8:11], map(float, high))))
    result.update(HF10_rad=float(np.sqrt(np.mean(high_frequency_rms(q, ENV.dt, 10) ** 2))),
        acceleration_rms_rad_s2=float(np.sqrt(np.mean(np.asarray(accelerations) ** 2))),
        torque_rate_rms_Nm_s=float(np.sqrt(np.mean((np.diff(tau, axis=0) / ENV.dt) ** 2))))
    return result


OBJECTIVE = {'roll_std_rad': 1.8, 'pitch_std_rad': 2.5, 'height_std_m': 2.0,
    'roll_rate_rms_rad_s': 1.0, 'pitch_rate_rms_rad_s': 1.5, 'vertical_speed_rms_m_s': 1.2,
    'roll_HF8_rad': .5, 'pitch_HF8_rad': .8, 'height_HF8_m': .8,
    'HF10_rad': 1.2, 'acceleration_rms_rad_s2': .3, 'torque_rate_rms_Nm_s': .3,
    'yaw_gait_rms_rad': .8, 'lateral_gait_rms_m': .8, 'cross_track_rmse_m': .3}


def score(cases, reference):
    ratios = {k: float(np.mean([c[k] for c in cases]) / max(1e-10, np.mean([r[k] for r in reference]))) for k in OBJECTIVE}
    violations = []
    for i, (c, r) in enumerate(zip(cases, reference)):
        if c['reason'] not in ('timeout', 'success'):
            violations.append((i, 'termination', 1.))
        checks = [(c['min_height_m'] >= .28, 'height'), (c['max_tilt_rad'] <= .08, 'tilt'),
            (c['height_peak_to_peak_m'] <= .008, 'height_range'),
            (c['nonfoot_contact_fraction'] == 0, 'contact'), (c['max_joint_speed_rad_s'] <= 20, 'speed'),
            (c['cross_track_rmse_m'] <= .05 and c['cross_track_max_m'] <= .15, 'tracking'),
            (c['progress_m'] >= .97 * r['progress_m'], 'progress'),
            (all(.08 <= v <= .55 for v in c['air_fraction']), 'stepping'),
            (all(v >= min(.029, .99 * ref) for v, ref in zip(c['feet_max_height_m'], r['feet_max_height_m'])), 'clearance')]
        violations.extend((i, name, 1.) for valid, name in checks if not valid)
        if r['reason'] == 'success' and (c['reason'] != 'success' or c['duration_s'] > 1.05 * r['duration_s']):
            violations.append((i, 'completion', 1.))
    limits = {'roll_std_rad': 1.04, 'pitch_std_rad': 1.04, 'height_std_m': 1.04,
              'HF10_rad': 1.05, 'yaw_gait_rms_rad': 1.08, 'lateral_gait_rms_m': 1.08}
    excess = sum(max(0., ratios[k] - limit) for k, limit in limits.items())
    cost = sum(OBJECTIVE[k] * ratios[k] ** 2 for k in OBJECTIVE)
    feasible = not violations and excess < 1e-10
    return {'score': -float(cost) if feasible else -1000. - 100 * len(violations) - 100 * excess - cost,
            'feasible': feasible, 'ratios': ratios, 'violations': violations, 'cases': cases,
            'steps': sum(c['steps'] for c in cases)}


def candidate(parameters):
    POLICY.weights['action_net.weight'] = weights(BASE, parameters)
    return score([measure_case(case) for case in CASES], REFERENCE)


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=6); parser.add_argument('--generations', type=int, default=7)
    parser.add_argument('--population', type=int, default=30); parser.add_argument('--seed', type=int, default=1022)
    parser.add_argument('--seconds', type=float, default=12); parser.add_argument('--skip-coordinate', action='store_true')
    parser.add_argument('--mean'); parser.add_argument('--full-paths', action='store_true')
    args = parser.parse_args(); out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        parser.error('Use an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.source).resolve(); model = NumpyPolicy(source); base = model.weights['action_net.weight'].copy()
    cfg = replace(Config(**model.pure_config), stage='path', seconds=args.seconds, reference_reward=False)
    cases = ([{'path': name, 'seconds': 60} for name in ['straight', 'sine', 'turn']] if args.full_paths else
        [{'path': 'straight', 'seconds': args.seconds}, {'path': 'sine', 'start_progress': .6, 'seconds': args.seconds},
         {'path': 'turn', 'start_progress': 1.65, 'seconds': args.seconds}, {'path': 'turn', 'start_progress': 2.7, 'seconds': args.seconds}])
    save_json(out / 'config.json', {'args': vars(args), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'environment': asdict(cfg), 'cases': cases, 'parameter_names': PARAMETER_NAMES, 'objective': OBJECTIVE,
        'measurement': 'At >=3 s, 500 Hz; body standard deviation, rate RMS and Welch >=8 Hz; joints Welch >=10 Hz',
        'model_sha256': hashlib.sha256((PROJECT / 'purequad/assets/quadruped.xml').read_bytes()).hexdigest()})
    with ProcessPoolExecutor(args.workers, initializer=initialize, initargs=(str(source), asdict(cfg), cases)) as pool:
        reference = list(pool.map(measure_case, cases))
    save_json(out / 'reference.json', reference)
    print(json.dumps({'event': 'baseline', 'metrics': reference}, ensure_ascii=False), flush=True)
    rng = np.random.default_rng(args.seed)
    std = np.array([.035] * 6 + [.025] * 3 + [.04] * 6 + [.06] * 6 + [.018] * 2 + [.012] * 3 + [.025] * 3 + [.01] * 3)
    assert len(std) == 32
    floor = std * .10; mean = np.zeros(32)
    if args.mean: mean = np.array(json.loads(Path(args.mean).read_text(encoding='utf-8')))
    bestp = np.zeros(32); best = score(reference, reference); steps = episodes = 0
    with ProcessPoolExecutor(args.workers, initializer=initialize, initargs=(str(source), asdict(cfg), cases, reference)) as pool:
        batches = ([-1] if not args.skip_coordinate else []) + list(range(args.generations))
        for generation in batches:
            if generation == -1:
                indices = list(range(6)) + list(range(9, 26))
                parameters = [np.zeros(32)]
                for index in indices:
                    for sign in [-1, 1]:
                        p = np.zeros(32); p[index] = sign * std[index] * 2; parameters.append(p)
                parameters = np.asarray(parameters)
            else:
                parameters = rng.normal(mean, std, (args.population, 32))
                parameters[0] = bestp; parameters[1] = np.zeros(32)
                if generation > 0: parameters[2] = bestp + rng.normal(0, std * .3)
            started = time.perf_counter(); rows = list(pool.map(candidate, parameters))
            scores = np.array([row['score'] for row in rows]); k = int(scores.argmax())
            steps += sum(r['steps'] for r in rows); episodes += len(rows) * len(cases)
            if rows[k]['score'] > best['score']:
                best = rows[k]; bestp = parameters[k].copy()
            model.weights['action_net.weight'] = weights(base, bestp)
            metadata = {'source': str(source), 'method': 'body_vibration_neural_weight_CEM', 'generation': generation,
                        'episodes': episodes, 'environment_steps': steps, 'parameters': bestp.tolist(), 'metrics': best}
            model.save(out / 'best_policy.zip', model.pure_config, metadata)
            model.save(out / f'generation_{generation:03d}.zip', model.pure_config, metadata)
            save_json(out / 'best_parameters.json', bestp.tolist()); save_json(out / 'best_candidate.json', metadata)
            with (out / 'generations.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps({'generation': generation, 'parameters': parameters.tolist(), 'candidates': rows}) + '\n')
            state = {'active': True, 'pid': os.getpid(), 'generation': generation, 'episodes': episodes, 'steps': steps,
                     'best_score': best['score'], 'feasible_candidates': sum(r['feasible'] for r in rows),
                     'ratios': best['ratios'], 'seconds': time.perf_counter() - started}
            save_json(out / 'training_state.json', state); print(json.dumps(state), flush=True)
            valid = [i for i in np.argsort(scores)[::-1] if rows[i]['feasible']]
            elite = parameters[valid[:max(4, args.population // 5)]] if valid else np.asarray([bestp])
            if generation == -1:
                mean = bestp.copy()
            else:
                mean = .25 * bestp + .75 * elite.mean(axis=0)
                std = np.maximum(.70 * std + .30 * elite.std(axis=0), floor)
            if (out / 'STOP').exists(): break
    state['active'] = False; save_json(out / 'training_state.json', state)


if __name__ == '__main__':
    main()

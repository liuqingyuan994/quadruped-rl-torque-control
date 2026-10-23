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

PROJECT = Path(os.environ.get('MINILITE_PROJECT', str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(PROJECT))
from purequad.config import Config
from purequad.env import QuadrupedEnv
from purequad.numpy_policy import NumpyPolicy
from purequad.paths import Path as Track
from purequad.metrics import high_frequency_rms

POLICY = ENV = BASE = BIAS = CASES = REFERENCE = None
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


def original_weights(base, parameters):
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


PARAMETER_NAMES += [f'clock{h+1}_{j}_phase' for h in range(3) for j in ['abduction','hip','knee']]
PARAMETER_NAMES += [f'front_rear_clock{h+1}_{j}_amplitude' for h in range(3) for j in ['hip','knee']]
PARAMETER_NAMES += [f'front_rear_clock{h+1}_{j}_phase' for h in range(3) for j in ['hip','knee']]
PARAMETER_NAMES += [f'{axis}_feedback_{j}' for axis in ['roll_rate','pitch_rate','vertical_velocity'] for j in ['hip','knee']]
PARAMETER_NAMES += [f'previous_action_{j}' for j in ['abduction','hip','knee']]
PARAMETER_NAMES += [f'learned_action_retention_{j}' for j in ['abduction','hip','knee']]

PARAMETER_NAMES += [f'{axis}_{j}' for axis in ['yaw_rate','heading','cross_track','lateral_velocity','lateral_velocity_add'] for j in ['abduction','hip','knee']]
PARAMETER_NAMES += [f'front_rear_abduction_clock{h+1}_{kind}' for kind in ['amplitude','phase'] for h in range(3)]

def weights(base, parameters, bias=None):
    p = np.asarray(parameters, dtype=float)
    if p.shape in ((62,), (65,)):p=np.pad(p,(0,86-len(p)))
    if p.shape != (86,) or not np.isfinite(p).all() or np.max(np.abs(p)) > .5 or np.min(p[62:65]) < 0:
        raise ValueError('Expected 86 finite parameters; retention must be in [0,.5]')
    if np.any(p[62:65]) and bias is None:raise ValueError('Action bias is required for retention weight editing')
    w = original_weights(base, p[:32])
    for leg in range(4):
        front = 1 if leg < 2 else -1
        for joint in range(3):
            row = 3*leg + joint
            for h in range(3):
                col = 124 + 2*h
                theta = p[32 + 3*h + joint]
                if joint in (1,2):
                    k = 2*h + joint-1
                    w[row,col:col+2] *= np.exp(front*p[41+k])
                    theta += front*p[47+k]
                a,b = w[row,col:col+2].copy()
                w[row,col] = a*np.cos(theta)-b*np.sin(theta)
                w[row,col+1] = a*np.sin(theta)+b*np.cos(theta)
            if joint in (1,2):
                for axis,col in enumerate([70,71,69]):
                    w[row,col] *= np.exp(p[53 + 2*axis + joint-1])
            w[row,97+row] *= np.exp(p[59+joint])
            retention=p[62+joint]
            if retention:
                # Fold the searched coefficients into the actor's existing last-action input.
                # obs[33:45] contains 0.4 * previous normalized action; obs[66] is constant 1.
                # Runtime inference, observations and direct torque writes are unchanged.
                w[row,:] *= 1-retention
                w[row,97+row] += retention/.4
                w[row,130] -= retention*bias[row]
    for leg in range(4):
        side=1 if leg in (0,2) else -1
        front=1 if leg<2 else -1
        for joint in range(3):
            row=3*leg+joint
            w[row,72] *= np.exp(p[65+joint])
            w[row,120] *= np.exp(p[68+joint]); w[row,137:143] *= np.exp(p[68+joint])
            w[row,119] *= np.exp(p[71+joint]); w[row,131:137] *= np.exp(p[71+joint])
            w[row,68] *= np.exp(p[74+joint])
            w[row,68] += side*p[77+joint]
        for h in range(3):
            row=3*leg; col=124+2*h
            w[row,col:col+2] *= np.exp(front*p[80+h])
            a,b=w[row,col:col+2].copy(); theta=front*p[83+h]
            w[row,col]=a*np.cos(theta)-b*np.sin(theta)
            w[row,col+1]=a*np.sin(theta)+b*np.cos(theta)
    return w


def initialize(source, configuration, cases, reference=None):
    global POLICY, ENV, BASE, BIAS, CASES, REFERENCE
    POLICY = NumpyPolicy(source)
    BASE = POLICY.weights['action_net.weight'].copy()
    BIAS = POLICY.weights['action_net.bias'].copy()
    assert POLICY.pure_config['action_scale'] == .4
    assert BASE.shape == (12, 143)
    np.testing.assert_array_equal(weights(BASE, np.zeros(62)), BASE)
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
    cycle_names = ['roll_cycle_rad','pitch_cycle_rad','height_cycle_m']
    if len(body) < 8:
        result.update({name: 1e3 for name in names + cycle_names})
        return result
    b = np.asarray(body); q = np.asarray(joints); tau = np.asarray(torques)
    std = np.std(b[:, :3], axis=0)
    rates = np.sqrt(np.mean((np.diff(b[:, :3], axis=0) / ENV.dt) ** 2, axis=0))
    high = high_frequency_rms(b[:, :3], ENV.dt, 8)
    result.update(dict(zip(cycle_names, map(float, high_frequency_rms(b[:,:3], ENV.dt, .7)))))
    result.update(dict(zip(names[:3], map(float, std))))
    result.update(yaw_gait_rms_rad=float(high_frequency_rms(b[:, 3:4], ENV.dt, .7)[0]),
        lateral_gait_rms_m=float(high_frequency_rms(b[:, 4:5], ENV.dt, .7)[0]))
    result.update(dict(zip(names[5:8], map(float, rates))))
    result.update(dict(zip(names[8:11], map(float, high))))
    result.update(HF10_rad=float(np.sqrt(np.mean(high_frequency_rms(q, ENV.dt, 10) ** 2))),
        acceleration_rms_rad_s2=float(np.sqrt(np.mean(np.asarray(accelerations) ** 2))),
        torque_rate_rms_Nm_s=float(np.sqrt(np.mean((np.diff(tau, axis=0) / ENV.dt) ** 2))))
    return result


OBJECTIVE = {'roll_cycle_rad': 4.0, 'pitch_cycle_rad': .8, 'height_cycle_m': .8,
    'roll_std_rad': .3, 'pitch_std_rad': .3, 'height_std_m': .3,
    'roll_rate_rms_rad_s': 1.0, 'pitch_rate_rms_rad_s': 1.5, 'vertical_speed_rms_m_s': 1.2,
    'roll_HF8_rad': .5, 'pitch_HF8_rad': .8, 'height_HF8_m': .8,
    'HF10_rad': 2.0, 'acceleration_rms_rad_s2': .5, 'torque_rate_rms_Nm_s': .5,
    'yaw_gait_rms_rad': 8.0, 'lateral_gait_rms_m': 5.0, 'cross_track_rmse_m': .3}


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
    limits = {'roll_cycle_rad': 1.02, 'pitch_cycle_rad': 1.02, 'height_cycle_m': 1.02,
              'roll_std_rad': 1.04, 'pitch_std_rad': 1.04, 'height_std_m': 1.04,
              'HF10_rad': 1.02, 'yaw_gait_rms_rad': 1.005, 'lateral_gait_rms_m': 1.005,
              'acceleration_rms_rad_s2': 1.03, 'torque_rate_rms_Nm_s': 1.03}
    excess = sum(max(0., ratios[k] - limit) for k, limit in limits.items())
    cost = sum(OBJECTIVE[k] * ratios[k] ** 2 for k in OBJECTIVE)
    feasible = not violations and excess < 1e-10
    return {'score': -float(cost) if feasible else -1000. - 100 * len(violations) - 100 * excess - cost,
            'feasible': feasible, 'ratios': ratios, 'violations': violations, 'cases': cases,
            'steps': sum(c['steps'] for c in cases)}


def candidate(parameters):
    POLICY.weights['action_net.weight'] = weights(BASE, parameters, BIAS)
    return score([measure_case(case) for case in CASES], REFERENCE)


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


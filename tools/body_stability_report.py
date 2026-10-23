"""Compare two independently evaluated frozen policies on all nine cases."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'tools/lateral_opt'))
from purequad.metrics import high_frequency_rms
from purequad.paths import Path as Track, wrap


def detailed(folder):
    result = json.loads((folder / 'acceptance.json').read_text(encoding='utf-8'))
    for case in result['cases']:
        data = pd.read_csv(folder / f"{case['path']}_{case['y_offset_m']:+.2f}" / 'trajectory.csv')
        data = data[data.time_s >= 3]
        body = data[['roll_rad', 'pitch_rad', 'z_m']].to_numpy()
        track = Track(case['path'])
        ix = np.clip(np.searchsorted(track.cumulative, data.progress_m, side='right') - 1, 0, len(track.delta) - 1)
        tangent = np.arctan2(track.delta[ix, 1], track.delta[ix, 0])
        heading = wrap(tangent - data.yaw_rad.to_numpy())
        std = np.std(body, axis=0)
        rates = np.sqrt(np.mean((np.diff(body, axis=0) / .002) ** 2, axis=0))
        high = high_frequency_rms(body, .002, 8)
        case.update(roll_std_rad=float(std[0]), pitch_std_rad=float(std[1]), height_std_m=float(std[2]),
            roll_rate_rms_rad_s=float(rates[0]), pitch_rate_rms_rad_s=float(rates[1]), vertical_speed_rms_m_s=float(rates[2]),
            roll_HF8_rad=float(high[0]), pitch_HF8_rad=float(high[1]), height_HF8_m=float(high[2]),
            yaw_gait_rms_rad=float(high_frequency_rms(heading[:, None], .002, .7)[0]),
            lateral_gait_rms_m=float(high_frequency_rms(data.cross_track_m.to_numpy()[:, None], .002, .7)[0]))
    return result


def compare(before, after):
    keys = ['roll_std_rad', 'pitch_std_rad', 'height_std_m', 'roll_rate_rms_rad_s', 'pitch_rate_rms_rad_s',
            'vertical_speed_rms_m_s', 'roll_HF8_rad', 'pitch_HF8_rad', 'height_HF8_m', 'yaw_gait_rms_rad',
            'lateral_gait_rms_m', 'joint_high_frequency_rms_rad', 'joint_acceleration_rms_rad_s2',
            'torque_rate_rms_Nm_s', 'height_peak_to_peak_m', 'duration_s', 'cross_track_rmse_m']
    metrics = {}
    for key in keys:
        a = float(np.mean([c[key] for c in before['cases']]))
        b = float(np.mean([c[key] for c in after['cases']]))
        metrics[key] = {'before': a, 'after': b, 'decrease_percent': 100 * (1 - b / a)}
    return {'before_sha256': before['sha256'], 'after_sha256': after['sha256'],
        'before_passed': sum(c['passed'] for c in before['cases']),
        'after_passed': sum(c['passed'] for c in after['cases']),
        'measurement': 'Equal mean of nine complete cases. Body and lateral/yaw >=3s; body HF>=8Hz, lateral/yaw HF>=0.7Hz; joint and other acceptance metrics use unchanged evaluate.py definitions.',
        'metrics': metrics, 'before_cases': before['cases'], 'after_cases': after['cases']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True); parser.add_argument('--after', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare(detailed(Path(args.before)), detailed(Path(args.after)))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('_cases')}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()

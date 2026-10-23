from pathlib import Path
import sys,json,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from purequad.metrics import high_frequency_rms
from purequad.paths import Path as Track,wrap

def detailed(folder):
    report=json.loads((folder/'acceptance.json').read_text(encoding='utf-8'))
    for c in report['cases']:
        d=pd.read_csv(folder/f"{c['path']}_{c['y_offset_m']:+.2f}"/'trajectory.csv');d=d[d.time_s>=3.]
        track=Track(c['path']);progress=d.progress_m.to_numpy()
        idx=np.clip(np.searchsorted(track.cumulative,progress,side='right')-1,0,len(track.delta)-1)
        tangent=np.arctan2(track.delta[idx,1],track.delta[idx,0]);head=wrap(tangent-d.yaw_rad.to_numpy())
        roll=d.roll_rad.to_numpy();cross=d.cross_track_m.to_numpy();pitch=d.pitch_rad.to_numpy()
        c.update(yaw_gait_rms_rad=float(high_frequency_rms(head[:,None],.002,cutoff=.7)[0]),
            lateral_gait_rms_m=float(high_frequency_rms(cross[:,None],.002,cutoff=.7)[0]),
            steady_roll_std_rad=float(np.std(roll)),steady_pitch_std_rad=float(np.std(pitch)),
            steady_roll_peak_to_peak_rad=float(np.ptp(roll)),steady_heading_peak_to_peak_rad=float(np.ptp(head)),
            steady_heading_std_rad=float(np.std(head)))
    return report

def compare(before,after):
    keys=['yaw_gait_rms_rad','lateral_gait_rms_m','steady_roll_std_rad','steady_pitch_std_rad',
        'steady_roll_peak_to_peak_rad','steady_heading_peak_to_peak_rad','steady_heading_std_rad',
        'joint_high_frequency_rms_rad','joint_acceleration_rms_rad_s2','torque_rate_rms_Nm_s',
        'height_peak_to_peak_m','duration_s','cross_track_rmse_m']
    result={'before_sha256':before['sha256'],'after_sha256':after['sha256'],'before_pass':sum(x['passed'] for x in before['cases']),
        'after_pass':sum(x['passed'] for x in after['cases']),
        'definition':'Equal-weight mean over 9 full path cases; lateral/yaw gait and steady attitude: >=3 s at 500 Hz, Welch >=0.7 Hz; original acceptance metrics unchanged'}
    for key in keys:
        x=float(np.mean([c[key] for c in before['cases']]));y=float(np.mean([c[key] for c in after['cases']]))
        result[key]={'before':x,'after':y,'decrease_percent':100*(1-y/x)}
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('after');p.add_argument('--output',required=True)
    p.add_argument('--baseline-json',default=str(Path(__file__).resolve().parents[2]/'results/mini_lite_urdf/lateral_opt_20261019/metrics_reference.json'))
    args=p.parse_args()
    before=json.loads(Path(args.baseline_json).read_text(encoding='utf-8'));after=detailed(Path(args.after));result=compare(before,after)
    Path(args.output).write_text(json.dumps({'comparison':result,'before_cases':before['cases'],'after_cases':after['cases']},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

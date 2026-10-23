"""Verify the newly frozen nine-case report against every recorded trajectory."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
PROJECT=Path('D:/Download/桌面/rl')
OUTPUT=PROJECT/'results/mini_lite_urdf/ppo_clearance_20261023'
POLICY=PROJECT/'runs/mini_lite_urdf/ppo_clearance_20261023/policy.zip'
ORIGINAL_REPORT=PROJECT/'results/mini_lite_urdf/ppo_curves_20261023/acceptance.json'
SOURCE_DATA=ROOT/'source_data';SOURCE_DATA.mkdir(exist_ok=True)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
r=json.loads((OUTPUT/'acceptance.json').read_text(encoding='utf-8'))
before=json.loads(ORIGINAL_REPORT.read_text(encoding='utf-8'))
assert r['all_passed'] and len(r['cases'])==9 and all(c['passed'] for c in r['cases'])
assert r['sha256']==sha(POLICY)==sha(ROOT/'selected_policy.zip')
assert r['criteria']==before['criteria']
assert [(c['path'],c['y_offset_m']) for c in r['cases']]==[(p,o) for p in ['straight','sine','turn'] for o in [0.,-.05,.05]]
sources={str(OUTPUT/'acceptance.json'):sha(OUTPUT/'acceptance.json'),str(ORIGINAL_REPORT):sha(ORIGINAL_REPORT)}
case_stats=[]
for c in r['cases']:
    name=f"{c['path']}_{c['y_offset_m']:+.2f}"
    csv=OUTPUT/name/'trajectory.csv'
    d=pd.read_csv(csv,encoding='utf-8-sig')
    sources[str(csv)]=sha(csv)
    assert len(d)==c['steps']
    # Contact columns are booleans; explicitly cast mixed numeric/bool columns.
    assert np.isfinite(d.to_numpy(dtype=np.float64)).all()
    assert np.allclose(np.diff(d.time_s),.002,atol=1e-10,rtol=0)
    assert np.isclose(d.time_s.iloc[-1],c['duration_s'],atol=1e-10)
    settled=d[d.time_s>=1]
    rmse=float(np.sqrt(np.mean(d.cross_track_m**2)))
    max_error=float(d.cross_track_m.abs().max())
    peaks=[float(settled[f'foot_{i}_height_m'].max()) for i in range(4)]
    air=[float((settled[f'contact_{i}']==0).mean()) for i in range(4)]
    assert np.isclose(rmse,c['cross_track_rmse_m'],atol=1e-12)
    assert np.isclose(max_error,c['cross_track_max_m'],atol=1e-12)
    np.testing.assert_allclose(peaks,c['feet_max_height_m'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(air,c['foot_air_fraction'],atol=1e-12,rtol=0)
    assert np.isclose(settled.z_m.max()-settled.z_m.min(),c['height_peak_to_peak_m'],atol=1e-12)
    assert min(peaks)>=.029 and all(.08<=v<=.55 for v in air)
    assert c['success'] and c['duration_s']<=60 and rmse<=.05 and max_error<=.15
    assert c['max_tilt_rad']<=.08 and c['min_height_m']>=.28 and c['height_peak_to_peak_m']<=.008
    assert c['nonfoot_contact_fraction']==0 and c['max_joint_speed_rad_s']<=20
    assert (OUTPUT/name/'motion.mp4').stat().st_size>10000
    case_stats.append({k:c[k] for k in ['path','y_offset_m','duration_s','cross_track_rmse_m','cross_track_max_m',
                      'feet_max_height_m','foot_air_fraction','max_tilt_rad','height_peak_to_peak_m','roll_rms_rad',
                      'pitch_rms_rad','joint_high_frequency_rms_rad','passed']})
    d[['time_s','x_m','y_m','progress_m','cross_track_m']].to_csv(SOURCE_DATA/(name+'.csv'),index=False,float_format='%.15g')
means={}
for key in ['cross_track_rmse_m','roll_rms_rad','pitch_rms_rad','height_peak_to_peak_m','joint_high_frequency_rms_rad']:
    a=float(np.mean([c[key] for c in before['cases']]))
    b=float(np.mean([c[key] for c in r['cases']]))
    means[key]={'before':a,'after':b,'ratio':b/a,'change_percent':(b/a-1)*100}
target=next(c for c in r['cases'] if c['path']=='straight' and c['y_offset_m']==-.05)
summary={'all_passed':True,'pass_count':9,'policy_sha256':r['sha256'],'criteria_unchanged':True,
         'target_RR_foot_peak_mm':target['feet_max_height_m'][3]*1000,
         'minimum_all_cases_foot_peak_mm':min(min(c['feet_max_height_m']) for c in r['cases'])*1000,
         'maximum_all_cases_duration_s':max(c['duration_s'] for c in r['cases']),
         'mean_comparisons':means,'cases':case_stats,'sources':sources,
         'videos_verified':9,'rows_analyzed':sum(c['steps'] for c in r['cases'])}
(ROOT/'final_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(SOURCE_DATA/'acceptance.json').write_bytes((OUTPUT/'acceptance.json').read_bytes())
print(json.dumps({k:v for k,v in summary.items() if k not in ['cases','sources']},ensure_ascii=False,indent=2))


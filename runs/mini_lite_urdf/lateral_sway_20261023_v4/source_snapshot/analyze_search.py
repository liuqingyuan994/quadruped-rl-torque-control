from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parent
folder=ROOT/sys.argv[1]
names=json.loads((folder/'config.json').read_text())['parameter_names']
rows=[json.loads(s) for s in (folder/'candidates.jsonl').read_text().splitlines()]
keys=['yaw_gait_rms_rad','lateral_gait_rms_m','roll_cycle_rad','pitch_cycle_rad','height_cycle_m','HF10_rad','cross_track_rmse_m']
def show(x):
    nonzero={names[i]:round(p,4) for i,p in enumerate(x['parameters']) if p}
    print(x['index'],round(x['metrics']['score'],3),nonzero if len(nonzero)<=12 else str(len(nonzero))+' edited axes','ratios', {k:round(x['metrics']['ratios'][k],3) for k in keys},'violations',x['metrics']['violations'])
print('Completed',len(rows),'feasible',sum(r['metrics']['feasible'] for r in rows))
for x in sorted(rows,key=lambda x:x['metrics']['score'],reverse=True)[:16]:show(x)
print('LOW YAW')
for x in sorted(rows,key=lambda x:x['metrics']['ratios']['yaw_gait_rms_rad'])[:12]:show(x)

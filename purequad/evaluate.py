import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,csv,json
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from .numpy_policy import NumpyPolicy
from .env import QuadrupedEnv
from .config import Config
from .camera import CameraTracker


def rollout(model,env,options=None,video=None,viewer=False,video_fps=60,camera_mode='stable'):
    env.camera_tracker=CameraTracker(camera_mode)
    obs,_=env.reset(seed=17,options=options);rows=[];total=0.;done=False;writer=None;handle=None;next_video_time=1/video_fps;next_viewer_time=1/video_fps
    if viewer:
        import mujoco.viewer,time
        handle=mujoco.viewer.launch_passive(env.model,env.data)
        start_wall=time.perf_counter()
        handle.cam.lookat[:]=env.data.qpos[:3]
        handle.cam.distance=1.5;handle.cam.azimuth=135;handle.cam.elevation=-25
        with handle.lock():
            handle.user_scn.ngeom=0;env.draw_path(handle.user_scn)
    if video:
        import imageio.v2 as io
        Path(video).parent.mkdir(parents=True,exist_ok=True)
        writer=io.get_writer(str(video),fps=video_fps,codec='libx264',quality=7,
            output_params=['-threads','1','-preset','veryfast','-tune','zerolatency'])
    try:
        while not done:
            action=model.predict(obs,deterministic=True)[0]
            obs,r,d,t,info=env.step(action);total+=r;done=d or t
            row={'time_s':info['time_s'],'x_m':info['position'][0],'y_m':info['position'][1],
                'z_m':info['position'][2],'roll_rad':info['rpy'][0],'pitch_rad':info['rpy'][1],'yaw_rad':info['rpy'][2],
                'progress_m':info['progress_m'],'cross_track_m':info['cross_track_m'],'reward':r,
                'joint_margin_rad':info['joint_margin_rad'],'nonfoot_contacts':info['nonfoot_contacts'],
                **{f'foot_{i}_height_m':info['feet_z'][i] for i in range(4)},
                **{f'contact_{i}':info['contacts'][i] for i in range(4)},
                **{f'torque_{i}_Nm':info['torque_Nm'][i] for i in range(12)},
                **{f'joint_{i}_rad':info['joint_position_rad'][i] for i in range(12)},
                **{f'joint_velocity_{i}_rad_s':info['joint_velocity_rad_s'][i] for i in range(12)},
                **{f'joint_acceleration_{i}_rad_s2':info['joint_acceleration_rad_s2'][i] for i in range(12)}}
            rows.append(row)
            if writer and info['time_s']+1e-10>=next_video_time:
                writer.append_data(env.render());next_video_time+=1/video_fps
            if handle and (info['time_s']+1e-10>=next_viewer_time or done):
                if not handle.is_running():info['reason']='viewer_closed';break
                if env.appearance is not None:env.appearance.update(env.data)
                target=env.camera_tracker.update(env.data.qpos[:3],info['time_s'])
                with handle.lock():handle.cam.lookat[:]=target
                time.sleep(max(0,start_wall+info['time_s']-time.perf_counter()))
                handle.sync();next_viewer_time+=1/video_fps
    finally:
        if writer:writer.close()
        if handle:handle.close()
    settled=[r for r in rows if r['time_s']>=1.] or rows
    feet_max=[max(r[f'foot_{i}_height_m'] for r in settled) for i in range(4)]
    air_fraction=[np.mean([not r[f'contact_{i}'] for r in settled]) for i in range(4)]
    summary={'path':env.path_name,'stage':env.cfg.stage,'duration_s':info['time_s'],'reason':info['reason'],
        'success':info['is_success'],'progress_m':info['progress_m'],'return':total,'steps':env.steps,
        'cross_track_rmse_m':float(np.sqrt(np.mean([r['cross_track_m']**2 for r in rows]))),
        'cross_track_max_m':max(abs(r['cross_track_m']) for r in rows),'min_height_m':min(r['z_m'] for r in rows),
        'height_peak_to_peak_m':float(np.ptp([r['z_m'] for r in settled])),
        'max_tilt_rad':max(max(abs(r['roll_rad']),abs(r['pitch_rad'])) for r in rows),
        'roll_rms_rad':float(np.sqrt(np.mean([r['roll_rad']**2 for r in settled]))),
        'min_joint_margin_rad':min(r['joint_margin_rad'] for r in rows),
        'nonfoot_contact_fraction':float(np.mean([r['nonfoot_contacts']>0 for r in rows])),
        'feet_max_height_m':feet_max,'foot_air_fraction':[float(v) for v in air_fraction],
        'architecture':'single neural policy to 12 full joint torques','reference_reward_enabled':env.cfg.reference_reward,
        'configuration':asdict(env.cfg)}
    q=np.array([[r[f'joint_{i}_rad'] for i in range(12)] for r in settled])
    dq=np.array([[r[f'joint_velocity_{i}_rad_s'] for i in range(12)] for r in settled])
    ddq=np.array([[r[f'joint_acceleration_{i}_rad_s2'] for i in range(12)] for r in settled])
    torque=np.array([[r[f'torque_{i}_Nm'] for i in range(12)] for r in settled])
    from .metrics import high_frequency_rms
    high_frequency=high_frequency_rms(q,env.dt)
    summary.update(max_joint_speed_rad_s=float(np.max(np.abs(dq))),joint_velocity_rms_rad_s=float(np.sqrt(np.mean(dq**2))),
        joint_acceleration_rms_rad_s2=float(np.sqrt(np.mean(ddq**2))),
        joint_high_frequency_rms_rad=float(np.sqrt(np.mean(high_frequency**2))),
        joint_high_frequency_per_joint_rad=high_frequency.tolist(),
        torque_rate_rms_Nm_s=float(np.sqrt(np.mean((np.diff(torque,axis=0)/env.dt)**2))) if len(torque)>1 else 0.,
        pitch_rms_rad=float(np.sqrt(np.mean([r['pitch_rad']**2 for r in settled]))),
        smoothness_definition='Samples after 1 s; joint position Welch PSD >=10 Hz; finite differences at control dt')
    return summary,rows


def save(output,summary,rows):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'trajectory.csv').open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='runs/mini_lite_urdf/final/policy.zip')
    p.add_argument('--path',choices=['straight','sine','turn'],default='straight')
    p.add_argument('--seconds',type=float,default=12);p.add_argument('--stage',choices=['stand','motion','path'])
    p.add_argument('--y-offset',type=float,default=0);p.add_argument('--start-progress',type=float)
    p.add_argument('--reference-reward',action='store_true');p.add_argument('--video');p.add_argument('--output',default='results/mini_lite_urdf/evaluation')
    p.add_argument('--viewer',action='store_true');p.add_argument('--fps',type=int,choices=range(1,501),default=60)
    p.add_argument('--camera',choices=['stable','raw'],default='stable')
    args=p.parse_args();model=NumpyPolicy(args.model)
    cfg=replace(Config(**model.pure_config),seconds=args.seconds,reference_reward=args.reference_reward)
    if args.stage:cfg.stage=args.stage
    env=QuadrupedEnv(cfg,args.path,visuals=bool(args.video or args.viewer));options={'y_offset':args.y_offset}
    if args.start_progress is not None:options['start_progress']=args.start_progress
    try:
        summary,rows=rollout(model,env,options,args.video,args.viewer,video_fps=args.fps,camera_mode=args.camera);summary['model']=args.model;summary['camera_mode']=args.camera;summary['display_fps']=args.fps;save(args.output,summary,rows)
        print(json.dumps({k:summary[k] for k in ['duration_s','reason','progress_m','return','min_height_m','max_tilt_rad','cross_track_rmse_m','feet_max_height_m','foot_air_fraction']},indent=2))
    finally:env.close()


if __name__=='__main__':main()

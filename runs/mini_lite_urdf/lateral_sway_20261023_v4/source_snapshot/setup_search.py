from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
PROJECT=Path('D:/Download/桌面/rl')
s=(PROJECT/'tools/body_cycle_core.py').read_text(encoding='utf-8')
s=s.replace("if p.shape == (62,):p=np.r_[p,np.zeros(3)]", "if p.shape in ((62,), (65,)):p=np.pad(p,(0,86-len(p)))")
s=s.replace("p.shape != (65,)","p.shape != (86,)").replace("np.min(p[62:])", "np.min(p[62:65])").replace("np.any(p[62:])", "np.any(p[62:65])").replace('Expected 65 finite parameters','Expected 86 finite parameters')
extra='''    for leg in range(4):
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
'''
needle='    return w\n\n\ndef initialize'
assert needle in s
s=s.replace(needle,extra+'\n\ndef initialize')
s=s.replace('\n\ndef weights(base, parameters, bias=None):', '''
PARAMETER_NAMES += [f'{axis}_{j}' for axis in ['yaw_rate','heading','cross_track','lateral_velocity','lateral_velocity_add'] for j in ['abduction','hip','knee']]
PARAMETER_NAMES += [f'front_rear_abduction_clock{h+1}_{kind}' for kind in ['amplitude','phase'] for h in range(3)]

def weights(base, parameters, bias=None):''')
s=s.replace("'yaw_gait_rms_rad': .8, 'lateral_gait_rms_m': .8", "'yaw_gait_rms_rad': 8.0, 'lateral_gait_rms_m': 5.0")
s=s.replace("'roll_cycle_rad': 2.5, 'pitch_cycle_rad': 2.5, 'height_cycle_m': 2.5", "'roll_cycle_rad': 4.0, 'pitch_cycle_rad': .8, 'height_cycle_m': .8")
s=s.replace("'HF10_rad': 1.005, 'yaw_gait_rms_rad': 1.03, 'lateral_gait_rms_m': 1.03", "'HF10_rad': 1.02, 'yaw_gait_rms_rad': 1.005, 'lateral_gait_rms_m': 1.005")
s=s.replace("'acceleration_rms_rad_s2': 1.02, 'torque_rate_rms_Nm_s': 1.02", "'acceleration_rms_rad_s2': 1.03, 'torque_rate_rms_Nm_s': 1.03")
(ROOT/'lateral_sway_core.py').write_text(s,encoding='utf-8')
s=(PROJECT/'tools/body_cycle_search.py').read_text(encoding='utf-8').replace('import body_cycle_core as search','import lateral_sway_core as search')
s=s.replace('import lateral_sway_core as search', 'sys.path.insert(0, str(Path(__file__).resolve().parent))\nimport lateral_sway_core as search')
s=s.replace('parameters.shape[1]==62:parameters=np.pad(parameters,((0,0),(0,3)))','parameters.shape[1] in (62,65):parameters=np.pad(parameters,((0,0),(0,86-parameters.shape[1])))')
s=s.replace('parameters.shape[1] != 65','parameters.shape[1] != 86').replace('N x 65','N x 86')
s=s.replace('body_cycle_and_joint_vibration_constrained_neural_search','lateral_yaw_roll_constrained_neural_search')
(ROOT/'lateral_sway_search.py').write_text(s,encoding='utf-8')
(ROOT/'lateral_sway_report.py').write_text((PROJECT/'tools/body_cycle_report.py').read_text(encoding='utf-8'),encoding='utf-8')
axes={0:.08,1:.04,2:.06,3:.04,4:.03,5:.03,6:.08,7:.03,8:.03,9:.10,10:.05,11:.06,12:.12,13:.08,14:.08,15:.18,17:.25,21:.025,22:.025,23:.02,24:.02,25:.02,26:.30,27:.15,28:.15,29:.05,30:.04,31:.04}
axes.update({i:.025 for i in range(32,41)})
axes.update({i:.03 for i in range(41,47)})
axes.update({i:.025 for i in range(47,53)})
axes.update({i:.25 for i in range(65,68)})
axes.update({i:.15 for i in range(68,74)})
axes.update({i:.20 for i in range(74,80)})
axes.update({i:.06 for i in range(80,83)})
axes.update({i:.025 for i in range(83,86)})
params=[[0.]*86]
for axis,delta in axes.items():
    for sign in [-1,1]:
        row=[0.]*86;row[axis]=sign*delta;params.append(row)
(ROOT/'probe_parameters.json').write_text(json.dumps(params),encoding='utf-8')
(ROOT/'probe_cases.json').write_text(json.dumps([{'path':'straight','seconds':12},{'path':'turn','start_progress':1.65,'seconds':12}]),encoding='utf-8')
print({'parameters':86,'candidates':len(params)})

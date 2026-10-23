"""Prepare a guarded release from independently accepted frozen policy files."""
from pathlib import Path
import hashlib,json,shutil,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
PROJECT=Path('D:/Download/桌面/rl')
TAG='lateral_sway_20261023_v4'
sys.path.insert(0,str(PROJECT))
from purequad.numpy_policy import NumpyPolicy

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    before=ROOT/'baseline_policy.zip';after=ROOT/'selected_policy.zip'
    comparison=read(ROOT/'comparison.json');acceptance=read(ROOT/'candidate_acceptance/acceptance.json')
    compatible=read(ROOT/'checkpoint_compatibility.json');interface=read(ROOT/'interface_check.json')
    transform=read(ROOT/'weight_transform_check.json');selected=read(ROOT/'selected_metadata.json')
    assert acceptance['all_passed'] and comparison['after_passed']==comparison['before_passed']==9
    assert digest(after)==acceptance['sha256']==comparison['after_sha256']
    assert digest(before)==comparison['before_sha256']==digest(PROJECT/'runs/mini_lite_urdf/final/policy.zip')
    assert compatible['SB3_checkpoint_load']=='PASS' and interface['checks_passed']==14 and transform['status']=='PASS'
    a,b=NumpyPolicy(before),NumpyPolicy(after)
    assert a.pure_config==b.pure_config
    changed=[k for k in a.weights if not np.array_equal(a.weights[k],b.weights[k])]
    assert changed==['action_net.weight']
    model_hash='81924c3eb55fae38c3a185c23393ef3ce9fb0a13dd5c33ec0f44e2b0973aa3b5'
    assert digest(PROJECT/'purequad/assets/quadruped.xml')==model_hash
    metrics=comparison['metrics']
    assert all(metrics[k]['after']<metrics[k]['before'] for k in ['yaw_gait_rms_rad','lateral_gait_rms_m','roll_cycle_rad'])
    assert metrics['joint_high_frequency_rms_rad']['after']<=1.02*metrics['joint_high_frequency_rms_rad']['before']
    assert all(metrics[k]['after'] <= 1.02*metrics[k]['before'] for k in ['pitch_cycle_rad','height_cycle_m'])
    release=ROOT/'release';release.mkdir(exist_ok=False);assets={}
    def copy(source,relative):
        target=release/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);assets[relative]=digest(target)
    def content(relative,text):
        target=release/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(text,encoding='utf-8');assets[relative]=digest(target)
    def data(relative,value):content(relative,json.dumps(value,ensure_ascii=False,indent=2))
    result=f'results/mini_lite_urdf/{TAG}';run=f'runs/mini_lite_urdf/{TAG}'
    for f in (ROOT/'candidate_acceptance').rglob('*'):
        if f.is_file():copy(f,f'{result}/final_acceptance/{f.relative_to(ROOT/"candidate_acceptance").as_posix()}')
    for name in ['comparison.json','checkpoint_compatibility.json','interface_check.json','weight_transform_check.json']:
        copy(ROOT/name,f'{result}/{name}')
    copy(ROOT/'baseline_acceptance/acceptance.json',f'{result}/baseline_acceptance.json')
    for f in ROOT.glob('trial_acceptance_*/acceptance.json'):copy(f,f'{result}/trials/{f.parent.name}.json')
    states=[]
    for f in sorted(ROOT.glob('0*/training_state.json')):
        state=read(f);assert not state['active'];states.append(state)
        for source in f.parent.iterdir():
            if source.is_file() and source.suffix in ['.json','.jsonl','.py']:
                copy(source,f'{run}/{f.parent.name}/{source.name}')
    for name in ['lateral_sway_core.py','lateral_sway_search.py','lateral_sway_report.py']:
        copy(ROOT/name,'tools/'+name);copy(ROOT/name,f'{run}/source_snapshot/{name}')
    for source in ROOT.glob('*.py'):
        if source.name not in ['lateral_sway_core.py','lateral_sway_search.py','lateral_sway_report.py']:copy(source,f'{run}/source_snapshot/{source.name}')
    copy(before,f'{run}/policy_before.zip');copy(after,f'{run}/policy_after.zip')
    source_hashes={p.relative_to(PROJECT).as_posix():digest(p) for p in (PROJECT/'purequad').glob('*.py')}
    selection={'method':'measured_lateral_yaw_roll_neural_weight_search',
        'before_sha256':digest(before),'after_sha256':digest(after),'selected':selected,
        'changed_tensors':changed,'configuration_changed':False,'model_sha256':model_hash,
        'fresh_baseline_acceptance':True,'fresh_acceptance':True,'source_hashes':source_hashes,
        'search_episodes':sum(s['episodes'] for s in states),'search_environment_steps':sum(s['steps'] for s in states),
        'simulation_only':True,'comparison':f'{result}/comparison.json','baseline_full_trajectories':str(ROOT/'baseline_acceptance')}
    data(f'{run}/selection.json',selection)
    guarded={}
    for name in ['README.md','project_status.json','runs/mini_lite_urdf/final/policy.zip',
        'runs/mini_lite_urdf/final/provenance.json','runs/mini_lite_urdf/final/selection.json','runs/mini_lite_urdf/final/publication_check.json']:
        guarded[name]=digest(PROJECT/name);backup=ROOT/'previous_project_metadata'/name
        backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/name,backup)
    report_path='docs/MiniLite左右摆动专项优化报告_20261023.md'
    video_path=f'videos/mini_lite_urdf/{TAG}/左右摆动专项优化_前后对照.mp4'
    status=read(PROJECT/'project_status.json')
    status.update(state='URDF_LATERAL_SWAY_ROUND4_ACCEPTANCE_PASS',policy_sha256=digest(after),
        all_passed=True,new_path_acceptance='PASS_9_OF_9',acceptance=f'{result}/final_acceptance/acceptance.json',
        latest_video=video_path,report=report_path,training_record=run,training_active=False,current_phase='complete',
        final_policy_method='Constrained lateral-yaw-roll neural actor weight optimization',
        lateral_sway_comparison={k:v for k,v in comparison.items() if not k.endswith('_cases')},post_publication_check='PENDING')
    status['body_stability_comparison']=status['lateral_sway_comparison']
    status['comparison']=f'{result}/comparison.json'
    data('project_status.json',status);data('runs/mini_lite_urdf/final/provenance.json',selection)
    data('runs/mini_lite_urdf/final/selection.json',{'source':f'{run}/policy_after.zip','sha256':digest(after),
        'acceptance':f'{result}/final_acceptance/acceptance.json','all_passed':True})
    data('runs/mini_lite_urdf/final/publication_check.json',{'status':'PRE_PUBLICATION_PASS','policy_sha256':digest(after),
        'mjcf_sha256':model_hash,'frozen_acceptance':'PASS_9_OF_9','fresh_baseline':'PASS_9_OF_9',
        'checkpoint_compatibility':compatible,'interface_checks':14,'changed_tensors':changed,'configuration_changed':False})
    pct=lambda k:metrics[k]['decrease_percent']
    joint_note=(f'关节高频振动（≥10 Hz）下降 {pct("joint_high_frequency_rms_rad"):.1f}%。'
        if abs(pct('joint_high_frequency_rms_rad'))>=.5 else '关节高频振动（≥10 Hz）基本持平。')
    headline=(f'相对上轮默认策略，九场景平均周期摆动（≥0.7 Hz）：航向下降 {pct("yaw_gait_rms_rad"):.1f}%，'
        f'横向平移下降 {pct("lateral_gait_rms_m"):.1f}%，横滚下降 {pct("roll_cycle_rad"):.1f}%；'
        +joint_note)
    old=(PROJECT/'README.md').read_text(encoding='utf-8');assert '\n## 运行' in old
    content('README.md','# Mini Lite 当前项目\n\n当前默认策略：2026-10-23 第四轮左右摆动专项优化，本轮重新冻结验收 **PASS 9/9**。\n\n'+headline+
        f'\n\n- [本轮优化报告]({report_path})\n- 验收及原始轨迹：`{result}/final_acceptance/`。\n'
        f'- 训练记录和上轮策略备份：`{run}/`。\n- 同视角对照视频：`{video_path}`。\n'
        '- 模型沿用用户提供的 URDF，质量 10.009390004 kg，2 ms 控制周期、67 维观测、12 个直接关节力矩动作。\n\n## 运行'+old.split('\n## 运行',1)[1])
    labels={'roll_cycle_rad':'横滚周期振动 rad','pitch_cycle_rad':'俯仰周期振动 rad','height_cycle_m':'上下周期振动 m',
        'roll_std_rad':'横滚标准差 rad','pitch_std_rad':'俯仰标准差 rad','height_std_m':'高度标准差 m',
        'roll_rate_rms_rad_s':'横滚速度 RMS rad/s','pitch_rate_rms_rad_s':'俯仰速度 RMS rad/s','vertical_speed_rms_m_s':'竖直速度 RMS m/s',
        'roll_HF8_rad':'横滚 ≥8 Hz 振动 rad','pitch_HF8_rad':'俯仰 ≥8 Hz 振动 rad','height_HF8_m':'高度 ≥8 Hz 振动 m',
        'yaw_gait_rms_rad':'航向周期摆动 rad','lateral_gait_rms_m':'横向周期摆动 m',
        'joint_high_frequency_rms_rad':'关节 ≥10 Hz 振动 rad','joint_acceleration_rms_rad_s2':'关节加速度 RMS rad/s²',
        'torque_rate_rms_Nm_s':'力矩变化率 RMS N·m/s','height_peak_to_peak_m':'高度峰峰值 m',
        'duration_s':'路径完成时间 s','cross_track_rmse_m':'路径 RMSE m'}
    report=['# Mini Lite 左右摆动专项优化 · 2026-10-23','',headline,'',
        '比较基线为第三轮默认策略 17913bcd…；优化前、优化后各自重新运行九场景。选择策略后冻结文件，再使用原 validate.py 验收，所有阈值保持一致。','',
        '本轮提高航向、横向平移与横滚周期摆动的优化权重，搜索各类关节的航向反馈、前后腿谐波幅值与相位，并约束俯仰、上下起伏、关节高频振动和行进速度。局部数值模型仅用于提出候选，所有选择均依据真实仿真结果。最终只改变 action_net.weight；物理模型、推理结构、观测维度、动作缩放和相机设置均保持一致。','',
        f'候选搜索共 {selection["search_episodes"]} 回合、{selection["search_environment_steps"]:,} 步；全部候选与失败记录随项目保存。','',
        '| 指标 | 第三轮基线 | 本轮 | 降低比例 |','| --- | ---: | ---: | ---: |']
    for key,label in labels.items():
        v=metrics[key];report.append(f'| {label} | {v["before"]:.6g} | {v["after"]:.6g} | {v["decrease_percent"]:.2f}% |')
    worse=[f'{label}增加 {-pct(k):.2f}%' for k,label in labels.items() if pct(k)<0]
    report += ['','负值表示增加。'+('本轮取舍：'+'；'.join(worse)+'。' if worse else '表内平均指标均改善。'),'',
        '统计口径：九场景等权平均；机身周期振动为 3 秒后的 500 Hz 样本、Welch 频谱 ≥0.7 Hz RMS；机身高频为 ≥8 Hz。关节高频为原 evaluate.py 中 1 秒后的 ≥10 Hz RMS。周期量、整体标准差和高频量分别统计，不相互替代。','',
        '| 路径 | 偏移 m | 完成时间 s | 最低足端峰值 m | 航向摆动降低 | 横向平移降低 | 横滚摆动降低 | 结果 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for a,b in zip(comparison['before_cases'],comparison['after_cases']):
        changes=[100*(1-b[k]/a[k]) for k in ['yaw_gait_rms_rad','lateral_gait_rms_m','roll_cycle_rad']]
        report.append(f'| {b["path"]} | {b["y_offset_m"]:+.2f} | {b["duration_s"]:.3f} | {min(b["feet_max_height_m"]):.5f} | {changes[0]:.2f}% | {changes[1]:.2f}% | {changes[2]:.2f}% | {"PASS" if b["passed"] else "FAIL"} |')
    per_case_increases=[100*(b['roll_std_rad']/a['roll_std_rad']-1) for a,b in zip(comparison['before_cases'],comparison['after_cases']) if b['roll_std_rad']>a['roll_std_rad']]
    if per_case_increases:report += ['',f'分场景限制：有 {len(per_case_increases)} 个曲线或转弯场景的整体横滚标准差仍略有增加，最大增加 {max(per_case_increases):.1f}%；周期摆动和整体标准差是不同指标，不能用前者的降幅替代后者。']
    report += ['',f'策略 SHA-256：`{digest(before)}` → `{digest(after)}`。',f'MJCF SHA-256：`{model_hash}`。','',
        '接口检查通过 14 组；检查点兼容性测试通过；神经权重变换的零修改恒等性、选中参数复现与输出层唯一变更检查通过。','',
        '对照录像为正常速度 60 FPS，直线、曲线、转弯各 12 秒；左侧为第三轮基线，右侧为本轮策略。','',
        f'上轮备份：`{run}/policy_before.zip`。关闭旧回放后重新运行 main.py，即加载更新后的默认策略。','',
        '验证范围为 MuJoCo 仿真，未执行实机测试。']
    content(report_path,'\n'.join(report)+'\n')
    video_check=read(ROOT/'videos/video_check.json');assert video_check['frames']==2160
    copy(ROOT/'videos/左右摆动专项优化_前后对照.mp4',video_path)
    copy(ROOT/'videos/video_check.json',f'{result}/video_check.json');copy(after,'runs/mini_lite_urdf/final/policy.zip')
    dump(ROOT/'release_manifest.json',{'project':str(PROJECT),'guarded':guarded,'files':assets,'new_policy_sha256':digest(after),
        'model_sha256':model_hash,'status':'READY_FOR_PUBLICATION'})
    print(json.dumps({'files':len(assets),'episodes':selection['search_episodes'],'steps':selection['search_environment_steps'],
        'bytes':sum((release/p).stat().st_size for p in assets)},ensure_ascii=False))

if __name__=='__main__':main()

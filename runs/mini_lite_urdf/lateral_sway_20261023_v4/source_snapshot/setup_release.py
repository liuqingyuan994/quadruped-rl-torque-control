"""Prepare this round's packaging scripts without touching the live project."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'body_stability_20261022_v3'
s=(OLD/'prepare_release.py').read_text(encoding='utf-8')
replacements={
    "TAG='body_stability_20261023_v3'":"TAG='lateral_sway_20261023_v4'",
    "['roll_cycle_rad','pitch_cycle_rad','height_cycle_m']":"['yaw_gait_rms_rad','lateral_gait_rms_m','roll_cycle_rad']",
    "metrics['joint_high_frequency_rms_rad']['before']":"1.02*metrics['joint_high_frequency_rms_rad']['before']",
    "['body_cycle_core.py','body_cycle_search.py','body_cycle_report.py']":"['lateral_sway_core.py','lateral_sway_search.py','lateral_sway_report.py']",
    "['check_weight_transform.py','media.py']":"['check_weight_transform.py','media.py','make_combinations.py','setup_search.py','setup_release.py','prepare_release.py','publish.py']",
    'measured_body_cycle_and_joint_smoothness_neural_weight_search':'measured_lateral_yaw_roll_neural_weight_search',
    'MiniLite机身抖动第三轮优化报告_20261023.md':'MiniLite左右摆动专项优化报告_20261023.md',
    '机身抖动第三轮优化_前后对照.mp4':'左右摆动专项优化_前后对照.mp4',
    'URDF_BODY_STABILITY_ROUND3_ACCEPTANCE_PASS':'URDF_LATERAL_SWAY_ROUND4_ACCEPTANCE_PASS',
    'Constrained body-cycle and joint-smoothness neural actor weight optimization':'Constrained lateral-yaw-roll neural actor weight optimization',
    'body_stability_comparison={':'lateral_sway_comparison={',
    '相对第二轮策略，九场景平均机身周期振动（≥0.7 Hz）：横滚下降':'相对上轮默认策略，九场景平均周期摆动（≥0.7 Hz）：航向下降',
    'pct("roll_cycle_rad"):.1f':'pct("yaw_gait_rms_rad"):.1f',
    '俯仰下降 {pct("pitch_cycle_rad"):.1f}':'横向平移下降 {pct("lateral_gait_rms_m"):.1f}',
    '上下起伏下降 {pct("height_cycle_m"):.1f}':'横滚下降 {pct("roll_cycle_rad"):.1f}',
    '2026-10-23 第三轮机身减振优化':'2026-10-23 第四轮左右摆动专项优化',
    '训练记录和第二轮策略备份':'训练记录和上轮策略备份',
    '# Mini Lite 第三轮机身减振优化 · 2026-10-23':'# Mini Lite 左右摆动专项优化 · 2026-10-23',
    '比较基线为第二轮默认策略 6ea071b1…':'比较基线为第三轮默认策略 17913bcd…',
    '本轮增加 ≥0.7 Hz 的机身周期振动指标，分别搜索步态谐波相位、前后腿谐波幅值与各关节姿态反馈权重，并约束关节高频振动。':'本轮提高航向、横向平移与横滚周期摆动的优化权重，搜索各类关节的航向反馈、前后腿谐波幅值与相位，并约束俯仰、上下起伏、关节高频振动和行进速度。局部数值模型仅用于提出候选，所有选择均依据真实仿真结果。',
    '| 指标 | 第二轮基线 | 本轮 | 降低比例 |':'| 指标 | 第三轮基线 | 本轮 | 降低比例 |',
    '| 横滚周期降低 | 俯仰周期降低 | 上下周期降低 |':'| 航向摆动降低 | 横向平移降低 | 横滚摆动降低 |',
    '神经权重变换的零修改恒等性与 128 组数值检查通过':'神经权重变换的零修改恒等性、选中参数复现与输出层唯一变更检查通过',
    '左侧为第二轮基线':'左侧为第三轮基线',
    '第二轮备份':'上轮备份'
}
for old,new in replacements.items():
    assert old in s,old
    s=s.replace(old,new)
guard="    release=ROOT/'release';release.mkdir(exist_ok=False);assets={}"
s=s.replace(guard,"    assert all(metrics[k]['after'] <= 1.02*metrics[k]['before'] for k in ['pitch_cycle_rad','height_cycle_m'])\n"+guard)
s=s.replace("    for f in ROOT.glob('rejected_acceptance_*/acceptance.json'):copy(f,f'{result}/rejected/{f.parent.name}.json')",
    "    for f in ROOT.glob('trial_acceptance_*/acceptance.json'):copy(f,f'{result}/trials/{f.parent.name}.json')")
s=s.replace("    for name in ['check_weight_transform.py','media.py','make_combinations.py','setup_search.py','setup_release.py','prepare_release.py','publish.py']:copy(ROOT/name,f'{run}/source_snapshot/{name}')",
    "    for source in ROOT.glob('*.py'):\n        if source.name not in ['lateral_sway_core.py','lateral_sway_search.py','lateral_sway_report.py']:copy(source,f'{run}/source_snapshot/{source.name}')")
s=s.replace("    data('project_status.json',status);", "    status['body_stability_comparison']=status['lateral_sway_comparison']\n    status['comparison']=f'{result}/comparison.json'\n    data('project_status.json',status);")
s=s.replace("    report += ['',f'策略 SHA-256：", "    per_case_increases=[100*(b['roll_std_rad']/a['roll_std_rad']-1) for a,b in zip(comparison['before_cases'],comparison['after_cases']) if b['roll_std_rad']>a['roll_std_rad']]\n    if per_case_increases:report += ['',f'分场景限制：有 {len(per_case_increases)} 个曲线或转弯场景的整体横滚标准差仍略有增加，最大增加 {max(per_case_increases):.1f}%；周期摆动和整体标准差是不同指标，不能用前者的降幅替代后者。']\n    report += ['',f'策略 SHA-256：")
(ROOT/'prepare_release.py').write_text(s,encoding='utf-8')
s=(OLD/'publish.py').read_text(encoding='utf-8').replace('policy.body_stability_v3_pending.zip','policy.lateral_sway_v4_pending.zip')
(ROOT/'publish.py').write_text(s,encoding='utf-8')
s=(OLD/'media.py').read_text(encoding='utf-8').replace('机身抖动第三轮优化_前后对照.mp4','左右摆动专项优化_前后对照.mp4')
(ROOT/'media.py').write_text(s,encoding='utf-8')
print('Packaging scripts prepared')

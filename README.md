# Mini Lite 当前项目

当前默认策略：2026-10-23 第四轮左右摆动专项优化，本轮重新冻结验收 **PASS 9/9**。

相对上轮默认策略，九场景平均周期摆动（≥0.7 Hz）：航向下降 6.2%，横向平移下降 2.5%，横滚下降 22.4%；关节高频振动（≥10 Hz）下降 2.3%。

- [本轮优化报告](docs/MiniLite左右摆动专项优化报告_20261023.md)
- 验收及原始轨迹：`results/mini_lite_urdf/lateral_sway_20261023_v4/final_acceptance/`。
- 训练记录和上轮策略备份：`runs/mini_lite_urdf/lateral_sway_20261023_v4/`。
- 同视角对照视频：`videos/mini_lite_urdf/lateral_sway_20261023_v4/左右摆动专项优化_前后对照.mp4`。
- 模型沿用用户提供的 URDF，质量 10.009390004 kg，2 ms 控制周期、67 维观测、12 个直接关节力矩动作。

## 运行

在此目录使用指定 Python：

```powershell
$py = 'D:/Software/anaconda3/envs/quadruped_rl/python.exe'
& $py -B -X utf8 main.py --task show --path sine
& $py -B -X utf8 main.py --task check
& $py -B -X utf8 main.py --task validate --workers 1 --output results/mini_lite_urdf/repeat_acceptance
```

相机为稳定模式、显示 60 FPS；物理控制仍为 500 Hz。重复验收使用新的输出目录，保留已冻结证据。

## 当前文件

- `purequad/`：模型加载、策略、训练、评估与验收代码；`assets/` 含当前模型及来源材料。
- `runs/mini_lite_urdf/final/`：当前策略与来源、选择和发布检查记录。
- `runs/mini_lite_urdf/training_record/`：上一版侧摆优化的训练配置、参考指标和代际日志。
- `results/mini_lite_urdf/lateral_opt_20261019/final_acceptance/`：上一版侧摆策略九场景验收及原始轨迹。
- `videos/mini_lite_urdf/lateral_opt_20261019/`：上一版侧摆策略对照与完整正弦路径录像。
- `tools/lateral_opt/`：完整路径侧摆优化及指标比较入口。
- [模型来源与参数](docs/模型来源与参数.md)、[侧摆优化报告](docs/MiniLite侧摆优化报告_20261019.md)。

## 继续训练

可用 `main.py --task train` 或 `main.py --task smooth` 从当前策略继续；从零训练用 `new_train.py --phase 1`，后续阶段使用新生成的前一阶段文件。

完整路径优化入口：

```powershell
& $py -B -X utf8 tools/lateral_opt/lateral_full_train.py --source runs/mini_lite_urdf/final/policy.zip --cases runs/mini_lite_urdf/training_record/full_cases.json --reference runs/mini_lite_urdf/training_record/full_baseline_training_metrics.json --output runs/mini_lite_urdf/next_lateral --workers 1
```

历史模型、旧策略、过期备份和训练中间文件已按用户要求清理。当前运行和继续训练不依赖被清理的检查点；训练来源中的旧路径仅作历史记录。原始提供文件仍位于 `E:/机器狗文件/`，未纳入本次删除范围。

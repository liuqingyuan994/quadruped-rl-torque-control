# PPO 训练后的抬脚约束优化

本策略从 262144 步 PPO 训练的最终权重出发，仅将后右腿髋屈伸、膝关节的六个周期特征权重乘以 1.01，共修改 12 个权重。
37 组候选单场景筛选、3 组完整九场景筛选后，冻结策略并通过项目原验收入口重新运行，9/9 全部通过。原门槛、物理模型与力矩缩放保持原设置。
source_snapshot 保存配置、候选指标、搜索代码、源策略与核对记录。该文件是独立实验策略，当前 final 默认策略保留。

在项目根目录的 PowerShell 中查看直线行走：

```powershell
conda activate quadruped_rl
python show_policy.py --model runs/mini_lite_urdf/ppo_clearance_20261023/policy.zip --path straight
```

结果：results/mini_lite_urdf/ppo_clearance_20261023/acceptance.json
图表：results/mini_lite_urdf/ppo_clearance_20261023/figures/

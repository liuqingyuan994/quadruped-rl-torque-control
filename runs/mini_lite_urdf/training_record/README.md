# 当前策略训练记录

保留最后一轮完整路径 CEM 的配置、奖励参考、代际日志与最优参数。当前默认策略在 ../final/policy.zip。
历史热启动模型与探索阶段文件已按用户要求清理；配置和日志里的旧 source 路径仅记录历史，不是运行依赖。
继续优化请以当前 final/policy.zip 为 --source，以本目录 full_cases.json 和 full_baseline_training_metrics.json 为 --cases、--reference，并指定新的 --output 目录。

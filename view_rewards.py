"""Export complete training episode returns by stage, without mixing validation episodes."""
import argparse,csv,json
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id',help='Only export this robot model.')
    parser.add_argument('--run-root',help='Only scan this training directory, relative to the project.')
    parser.add_argument('--output',default='results/rewards')
    args=parser.parse_args()
    rows=[];groups=[]
    directories=set()
    parents=[ROOT/args.run_root] if args.run_root else [ROOT/'logs/training',ROOT/'runs']
    for parent in parents:
        if parent.exists():
            for pattern in ('generations.jsonl','env_*.monitor.csv'):
                directories.update(p.parent for p in parent.rglob(pattern))
    for run in sorted(directories):
        configuration=json.loads((run/'config.json').read_text(encoding='utf-8')) if (run/'config.json').exists() else {}
        robot_model=configuration.get('environment',{}).get('robot_model','legacy_paper_parameter_quadruped')
        if args.model_id and robot_model!=args.model_id:continue
        history=run/'generations.jsonl'
        if history.exists():
            start=len(rows);number=0
            for line in history.read_text(encoding='utf-8').splitlines():
                if not line.strip():continue
                g=json.loads(line)
                for candidate,entry in enumerate(g['candidates'],1):
                    for case,c in enumerate(entry.get('cases',[entry]),1):
                        number+=1;rows.append({'robot_model':robot_model,'stage':run.relative_to(ROOT).as_posix(),'method':'CEM','episode':number,'return':c.get('raw_return',c['return']),'search_score':c['return'],
                            'steps':c['steps'],'reason':c['reason'],'generation':g['generation']+1,'candidate':candidate,'case':case})
            groups.append((run.relative_to(ROOT).as_posix(),rows[start:]))
        monitor=list(run.glob('env_*.monitor.csv'))
        if monitor:
            data=[]
            for f in monitor:
                with f.open(encoding='utf-8') as h:lines=h.readlines()
                meta=json.loads(lines[0][1:])
                for r in csv.DictReader(x for x in lines if not x.startswith('#')):
                    if r.get('l'):data.append((float(meta['t_start'])+float(r['t']),r))
            data.sort(key=lambda x:x[0]);start=len(rows)
            for i,(_,r) in enumerate(data,1):
                rows.append({'robot_model':robot_model,'stage':run.relative_to(ROOT).as_posix(),'method':'PPO','episode':i,'return':float(r['r']),
                    'steps':int(r['l']),'reason':r['reason'],'generation':'','candidate':'','case':'','search_score':''})
            groups.append((run.relative_to(ROOT).as_posix(),rows[start:]))
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    if not rows:raise RuntimeError('No completed training episodes in logs/training or runs')
    with (out/'episode_rewards.csv').open('w',newline='',encoding='utf-8-sig') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(groups),1,figsize=(11,2.5*len(groups)),constrained_layout=True,squeeze=False)
    for ax,(name,records) in zip(axes[:,0],groups):
        ax.plot([r['episode'] for r in records],[r['return'] for r in records],'.',markersize=2,color='#22699c')
        ax.set(title=name,xlabel='Complete episode',ylabel='Episode return');ax.grid(alpha=.2)
    fig.savefig(out/'reward_curves.png',dpi=130);plt.close(fig)
    (out/'training_rewards.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>纯 RL 每回合奖励</title><body style="max-width:1000px;margin:30px auto;font:18px/1.8 Microsoft YaHei"><h1>纯强化学习：每回合累计奖励</h1><p>日志位于 logs/training，新训练日志从 runs 自动合并。各阶段奖励版本和回合长度不同，不直接比较总奖励。只统计完整训练回合，不包含验收和录像。页面可能包含旧机器人训练；CSV 的 robot_model 列区分模型，不能把旧记录作为当前 Mini Lite 的训练结果。</p><p>完整回合总数：'+str(len(rows))+'</p><p><a href="episode_rewards.csv">下载逐回合 CSV</a></p><img style="width:100%" src="reward_curves.png"><p>return 是每步环境奖励相加的原始回报。CEM 若额外扣除抬脚或前进不足罚分，则 search_score 保存扣分后的搜索评分；两列含义不同。历史旧阶段没有额外扣分时两列相同。PPO Monitor 的 r 列为原始回报。</p></body></html>',encoding='utf-8')
    print('Exported',len(rows),'complete training episodes to',out)


if __name__=='__main__':main()

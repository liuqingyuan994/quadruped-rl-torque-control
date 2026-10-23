"""以用户提供 Mini Lite URDF 为准的纯关节力矩强化学习入口。"""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
MODEL='runs/mini_lite_urdf/final/policy.zip'


def run(task,arguments):
    arguments=list(arguments)
    headless='--headless' in arguments
    if headless:arguments.remove('--headless')
    python=json.loads((ROOT/'python_environment.json').read_text(encoding='utf-8'))['python_executable']
    if task in ('show','record','evaluate','validate') and not (ROOT/MODEL).is_file() and '--model' not in arguments:
        print('Mini Lite 尚无新训练的最终模型。先用 new_train.py --phase 1 开始训练，或用 --model 指定本模型的新策略。');return 1
    if task in ('train','smooth') and not (ROOT/MODEL).is_file() and '--source' not in arguments:
        print('继续训练需要本模型的新策略。先用 new_train.py --phase 1 从零训练，或用 --source 指定新策略。');return 1
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--path',choices=['straight','sine','turn'],default='turn')
    path=parser.parse_known_args(arguments)[0].path
    defaults={
        'start':('new_train',['--phase','1']),
        'check':('purequad.check',[]),
        'show':('purequad.evaluate',['--model',MODEL,'--stage','path','--path',path,'--seconds','90','--viewer']),
        'record':('purequad.evaluate',['--model',MODEL,'--stage','path','--path',path,'--seconds','90','--video',f'videos/mini_lite_urdf/{path}/motion.mp4','--output',f'videos/mini_lite_urdf/{path}']),
        'evaluate':('purequad.evaluate',['--model',MODEL,'--stage','path','--seconds','90','--output','results/mini_lite_urdf/evaluation']),
        'train':('purequad.train',['--source',MODEL,'--stage','path','--output','runs/mini_lite_urdf/my_ppo']),
        'validate':('purequad.validate',['--model',MODEL,'--output','results/mini_lite_urdf/acceptance']),
        'smooth':('purequad.smooth_train',['--source',MODEL,'--workers','1','--output','runs/mini_lite_urdf/my_smoothing'])}
    module,args=defaults[task]
    if headless and '--viewer' in args:args.remove('--viewer')
    environment=os.environ.copy()
    environment.update(PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    return subprocess.call([python,'-X','utf8','-m',module,*args,*arguments],cwd=ROOT,env=environment)


def main():
    tasks={'start':'从零开始 Mini Lite 站立训练','check':'检查 URDF 参数与纯力矩接口',
           'show':'显示新策略','record':'录制新策略','evaluate':'无窗口评估新策略',
           'train':'继续 PPO 训练','validate':'新模型九场景验收','smooth':'继续关节平稳性 CEM'}
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=tasks)
    args,remaining=parser.parse_known_args()
    if args.task:return run(args.task,remaining)
    while True:
        print('\nMini Lite · URDF 参数 · 12 个完整关节力矩')
        for i,label in enumerate(tasks.values(),1):print(f'{i}. {label}')
        print('0. 退出')
        try:answer=input('请选择：').strip()
        except (EOFError,KeyboardInterrupt):return 0
        if answer=='0':return 0
        if answer.isdigit() and 1<=int(answer)<=len(tasks):run(list(tasks)[int(answer)-1],[])


if __name__=='__main__':raise SystemExit(main())

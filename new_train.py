"""Copy into the project root. Run one curriculum phase at a time with this Python."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def recipe(phase):
    root = Path('runs/mini_lite_urdf')
    output = str(root / f'{phase:02d}')
    if phase == 7:
        return 'purequad.smooth_train', [
            '--source', str(root / '06/best_policy.zip'), '--output', output,
            '--workers', '1', '--generations', '13', '--population', '32',
            '--seconds', '6', '--seed', '81'], output
    if phase == 4:
        return 'purequad.train', [
            '--source', str(root / '03/best_policy.zip'), '--output', output,
            '--stage', 'motion',  '--steps', '300000',
            '--envs', '4', '--log-std', '-3.7', '--eval-freq', '50000',
            '--eval-seconds', '15', '--seed', '61'], output
    specifications = {
        1: ('stand', None, 40, 96, 4, 51, 1, 1),
        2: ('motion', '01/best_policy.zip', 50, 64, 6, 53, 1, 3),
        3: ('feedback', '02/best_policy.zip', 35, 64, 8, 59, .7, 3),
        5: ('balance', '04/best_policy.zip', 28, 64, 8, 63, 1, 3),
        6: ('steer', '05/best_policy.zip', 32, 64, 7, 67, 1, 3),
    }
    mode, source, generations, population, seconds, seed, sigma, reward = specifications[phase]
    arguments = ['--mode', mode, '--output', output, '--workers', '1',
                 '--generations', str(generations), '--population', str(population),
                 '--seconds', str(seconds), '--seed', str(seed),
                 '--sigma', str(sigma), '--reward-version', str(reward)]
    if source:
        arguments += ['--source', str(root / source)]
    if phase == 6:
        arguments += ['--options', json.dumps([
            {'path': 'straight', 'start_progress': 0, 'y_offset': .05},
            {'path': 'sine', 'start_progress': .2, 'y_offset': 0},
            {'path': 'turn', 'start_progress': 1.65, 'y_offset': 0},
            {'path': 'turn', 'start_progress': 2.7, 'y_offset': 0},
        ])]
    return 'purequad.cem', arguments, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', type=int, choices=range(1, 8), required=True)
    parser.add_argument('--source', help='Override the previous phase candidate; forbidden for phase 1.')
    parser.add_argument('--output', help='Use a new empty directory to repeat a phase.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.phase == 1 and args.source:
        parser.error('Phase 1 starts without a checkpoint. Do not specify --source.')
    module, arguments, output = recipe(args.phase)
    if args.source:
        arguments[arguments.index('--source') + 1] = args.source
    if args.output:
        output = args.output
        arguments[arguments.index('--output') + 1] = output
    command = [sys.executable, '-X', 'utf8', '-m', module, *arguments]
    print(subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return 0
    if not (ROOT / 'purequad').is_dir():
        parser.error('Copy this script into the project root, beside main.py.')
    if (ROOT / output).exists() and any((ROOT / output).iterdir()):
        parser.error('Output directory is not empty; choose a new --output to keep logs separate.')
    if '--source' in arguments:
        source = arguments[arguments.index('--source') + 1]
        if not (ROOT / source).is_file():
            parser.error(f'Previous phase model is missing: {source}')
    environment = os.environ.copy()
    environment.update(PYTHONUTF8='1', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return subprocess.call(command, cwd=ROOT, env=environment)


if __name__ == '__main__':
    raise SystemExit(main())

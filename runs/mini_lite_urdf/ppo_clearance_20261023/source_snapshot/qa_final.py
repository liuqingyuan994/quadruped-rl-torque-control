"""Audit the final plotting source and exported PDF."""
from pathlib import Path
import json
import subprocess

ROOT=Path(__file__).resolve().parent
SCRIPTS=Path('C:/Users/27104/.codex/skills/nature-figure/scripts')
PYTHON='D:/Software/anaconda3/envs/quadruped_rl/python.exe'
QA=ROOT/'qa';QA.mkdir(exist_ok=True)
def call(name,*arguments):
    return subprocess.run([PYTHON,'-B','-X','utf8',str(SCRIPTS/name),*map(str,arguments)],
                          capture_output=True,text=True,encoding='utf-8')
source=call('validate_figure.py',ROOT/'plot_final.py','--json')
(QA/'source.json').write_text(source.stdout,encoding='utf-8')
pdf=ROOT/'figures/ppo_clearance_tracking_errors.pdf'
glyph=call('audit_pdf_text.py',pdf,'--min-pt','5','--json')
(QA/'glyphs.json').write_text(glyph.stdout,encoding='utf-8')
collisions=call('audit_figure_collisions.py',pdf,'--json-out',QA/'collisions.json','--overlay-pdf',QA/'collision-overlay.pdf')
(QA/'collisions.log').write_text(collisions.stdout+collisions.stderr,encoding='utf-8')
verdict=json.loads((QA/'collisions.json').read_text(encoding='utf-8'))['verdict']
result={'source_exit':source.returncode,'glyph_exit':glyph.returncode,'collision_exit':collisions.returncode,
        'collision_verdict':verdict,'source_warning_review':'PNG is used for Obsidian; no TIFF journal output is requested.'}
(QA/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
assert source.returncode==0 and glyph.returncode==0 and collisions.returncode==0 and verdict=='PASS'

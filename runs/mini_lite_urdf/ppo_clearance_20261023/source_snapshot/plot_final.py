"""Plot every recorded sample of the accepted frozen policy using Python."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent
FIGURES=ROOT/'figures';FIGURES.mkdir(exist_ok=True)
SCRIPTS=Path('C:/Users/27104/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
    'font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
    'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
    'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
assert json.loads((ROOT/'final_analysis.json').read_text(encoding='utf-8'))['all_passed']
fig,axes=plt.subplots(3,1,figsize=(7.2047244,6.8897638))
fig.subplots_adjust(left=.12,right=.97,bottom=.075,top=.90,hspace=.68)
fig.suptitle('PPO 训练与抬脚微调后：九场景横向跟踪误差',y=.978,fontsize=10)
labels={'straight':'直线路径','sine':'曲线路径','turn':'转弯路径'}
for index,(ax,path) in enumerate(zip(axes,['straight','sine','turn'])):
    ax.set_title(labels[path],loc='left',pad=9)
    ax.text(-.10,1.07,'abc'[index],transform=ax.transAxes,ha='left',va='bottom',fontsize=9,fontweight='bold')
    ax.grid(axis='y',color='#DDE2E6',linewidth=.45,zorder=0)
    ax.tick_params(direction='out',length=3,width=.6)
    for color,offset,style in zip(['#537A95','#C48A5A','#6B8B7E'],[0.,-.05,.05],['-','--',':']):
        data=pd.read_csv(ROOT/'source_data'/f'{path}_{offset:+.2f}.csv')
        assert np.isfinite(data.to_numpy()).all()
        assert (np.diff(data.time_s)>0).all()
        ax.plot(data.time_s,data.cross_track_m*100,color=color,linestyle=style,linewidth=.8,
                label=f'起始偏移 {offset*100:+.0f} cm')
    ax.axhline(0,color='#7F8790',linewidth=.5,zorder=0)
    ax.set_xlim(0,50);ax.set_ylim(-5.6,5.6);ax.set_yticks([-5,0,5])
    ax.set_xlabel('仿真时间 (s)');ax.set_ylabel('横向误差 (cm)')
    if index==0:
        ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.34),frameon=False,columnspacing=2.1,handlelength=2.5)
name='ppo_clearance_tracking_errors'
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=FIGURES/(name+'.alignment.json'),
    overlay_svg=FIGURES/(name+'.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,
    require_panel_labels=True,strict=True)
fig.savefig(FIGURES/(name+'.png'),dpi=600)
fig.savefig(FIGURES/(name+'.svg'))
fig.savefig(FIGURES/(name+'.pdf'))
plt.close(fig)
print('Exported accepted-policy tracking figure: PNG, SVG, PDF.')

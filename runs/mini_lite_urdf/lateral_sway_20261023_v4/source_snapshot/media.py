"""Record and compare two policies at the same real-time playback speed."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import imageio_ffmpeg


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--before',required=True)
    p.add_argument('--after',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    for scene,start in [('straight',0),('sine',0),('turn',1.65)]:
        for label,policy in [('before',args.before),('after',args.after)]:
            subprocess.run([sys.executable,'-B','-X','utf8','-u','-m','purequad.evaluate',
                '--model',policy,'--stage','path','--path',scene,'--seconds','12','--fps','60','--camera','stable',
                '--start-progress',str(start),'--video',str(root/f'{label}_{scene}.mp4'),
                '--output',str(root/f'{label}_{scene}')],cwd=args.project,check=True)
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();font='C\\:/Windows/Fonts/msyh.ttc'
    for scene,cn in [('straight','直线'),('sine','曲线'),('turn','转弯')]:
        filt=(f"[0:v][1:v]hstack=inputs=2,drawbox=x=0:y=0:w=iw:h=44:color=black@0.65:t=fill,"
            f"drawtext=fontfile='{font}':text='上轮版本 · {cn}':x=18:y=8:fontsize=25:fontcolor=white,"
            f"drawtext=fontfile='{font}':text='本轮优化 · {cn}':x=658:y=8:fontsize=25:fontcolor=white[v]")
        subprocess.run([ffmpeg,'-v','error','-y','-i',str(root/f'before_{scene}.mp4'),'-i',str(root/f'after_{scene}.mp4'),
            '-filter_complex',filt,'-map','[v]','-c:v','libx264','-preset','fast','-crf','20','-threads','2',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(root/f'compare_{scene}.mp4')],check=True)
    target=root/'左右摆动专项优化_前后对照.mp4'
    subprocess.run([ffmpeg,'-v','error','-y','-i',str(root/'compare_straight.mp4'),'-i',str(root/'compare_sine.mp4'),'-i',str(root/'compare_turn.mp4'),
        '-filter_complex','[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]','-map','[v]','-c:v','libx264','-preset','fast','-crf','20',
        '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
    frames=imageio_ffmpeg.read_frames(str(target));meta=next(frames);count=sum(1 for _ in frames)
    assert count==2160 and tuple(meta['size'])==(1280,480) and meta['fps']==60
    (root/'video_check.json').write_text(json.dumps({'frames':count,'meta':meta,'before':args.before,
        'after':args.after,'scene_duration_s':12,'normal_playback_speed':True,'camera':'stable'},indent=2),encoding='utf-8')
    subprocess.run([ffmpeg,'-v','error','-y','-ss','6','-i',str(target),'-frames:v','1',str(root/'comparison_preview.png')],check=True)
    print(json.dumps({'video':str(target),'frames':count,'fps':meta['fps']}),flush=True)


if __name__=='__main__':main()

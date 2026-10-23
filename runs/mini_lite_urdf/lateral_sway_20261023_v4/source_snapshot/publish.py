"""Publish only manifest-listed files, then verify the actual default entry point."""
from pathlib import Path
from dataclasses import replace
import hashlib
import json
import os
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=Path('D:/Download/桌面/rl').resolve()

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    manifest=read(ROOT/'release_manifest.json')
    assert Path(manifest['project']).resolve()==PROJECT
    for name,sha in manifest['guarded'].items():
        assert digest(PROJECT/name)==sha, f'Concurrent modification: {name}'
        assert digest(ROOT/'previous_project_metadata'/name)==sha
    for name,sha in manifest['files'].items():
        target=(PROJECT/name).resolve();source=(ROOT/'release'/name).resolve()
        assert PROJECT in target.parents and (ROOT/'release').resolve() in source.parents
        assert digest(source)==sha
        assert not target.exists() or name in manifest['guarded'], f'Unapproved overwrite: {name}'
    policy_name='runs/mini_lite_urdf/final/policy.zip'
    try:
        for name in manifest['files']:
            if name==policy_name:continue
            target=PROJECT/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/'release'/name,target)
        target=PROJECT/policy_name;temp=target.with_name('policy.lateral_sway_v4_pending.zip')
        assert not temp.exists();shutil.copy2(ROOT/'release'/policy_name,temp);os.replace(temp,target)
        for name,sha in manifest['files'].items():assert digest(PROJECT/name)==sha,name
        sys.path.insert(0,str(PROJECT))
        import main as entry
        from purequad.config import Config
        from purequad.env import QuadrupedEnv
        from purequad.numpy_policy import NumpyPolicy
        assert entry.ROOT.resolve()==PROJECT
        filename=entry.ROOT/entry.MODEL;assert digest(filename)==manifest['new_policy_sha256']
        policy=NumpyPolicy(filename);env=QuadrupedEnv(replace(Config(**policy.pure_config),seconds=2), 'sine')
        try:
            obs,_=env.reset(seed=17)
            for step in range(500):
                assert obs.shape==(67,) and np.isfinite(obs).all()
                action=policy.predict(obs)[0];assert action.shape==(12,) and np.isfinite(action).all()
                obs,reward,terminated,truncated,info=env.step(action)
                assert not terminated and not truncated,info
        finally:env.close()
        status=read(PROJECT/'project_status.json')
        accepted=read(PROJECT/status['acceptance'])
        assert status['policy_sha256']==accepted['sha256']==digest(filename)
        assert accepted['all_passed'] and not status['training_active']
        assert digest(PROJECT/'purequad/assets/quadruped.xml')==manifest['model_sha256']
        replay={'status':'PASS','policy_sha256':digest(filename),'steps':500,'observations':67,'actions':12,
            'default_entry':str(entry.ROOT/'main.py'),'model_sha256':manifest['model_sha256']}
        dump(ROOT/'post_publication_replay.json',replay)
        status['post_publication_check']='PASS';dump(PROJECT/'project_status.json',status)
        pub=read(PROJECT/'runs/mini_lite_urdf/final/publication_check.json')
        pub.update(status='PASS',published_files_verified=len(manifest['files']),default_entry_replay=replay)
        dump(PROJECT/'runs/mini_lite_urdf/final/publication_check.json',pub)
        dump(ROOT/'publication_receipt.json',{'status':'PASS','verified_files':len(manifest['files']),
            'policy_sha256':manifest['new_policy_sha256'],'default_entry_replay':replay})
        print(json.dumps(replay,ensure_ascii=False))
    except Exception:
        for name in manifest['guarded']:shutil.copy2(ROOT/'previous_project_metadata'/name,PROJECT/name)
        dump(ROOT/'publication_receipt.json',{'status':'FAILED_DEFAULT_RESTORED'})
        raise

if __name__=='__main__':main()

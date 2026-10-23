"""Validate supplied URDF tensors, kinematics and the direct torque interface."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .env import QuadrupedEnv
from .config import Config
from .model import MODEL_ID,ASSETS,SOURCE,LEGS,SDK_LEGS,sha256,assert_policy_compatible
from .numpy_policy import NumpyPolicy


def rotation(axis,angle):
    x,y,z=axis/np.linalg.norm(axis)
    k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)


def run_checks():
    manifest=json.loads((ASSETS/'robot_parameters.json').read_text(encoding='utf-8'))
    urdf=ET.parse(SOURCE/'mini_lite/urdf/MiniLiteC.runtime.urdf').getroot()
    source=ET.parse(SOURCE/'mini_lite/urdf/MiniLiteC.urdf').getroot()
    source.find("link[@name='HR_HIP']/inertial/inertia").set('izz','0.00014559')
    def structure(n):return n.tag,n.attrib,[structure(c) for c in n]
    assert structure(source)==structure(urdf)
    assert sha256(SOURCE/'mini_lite/urdf/MiniLiteC.urdf')==manifest['source_urdf_sha256']
    a,b=QuadrupedEnv(),QuadrupedEnv(visuals=True);checks=[]
    try:
        assert (a.model.nq,a.model.nv,a.model.nu)==(19,18,12)
        assert (a.model.nmesh,b.model.nmesh)==(0,8)
        assert a.observation_space.shape==(67,) and a.action_space.shape==(12,)
        assert abs(a.model.body_mass.sum()-manifest['total_mass_kg'])<1e-12
        assert abs(a.model.body_mass[a.torso]-5.6352)<1e-12
        assert np.array_equal(a.limits,np.tile([13.2,13.2,28],4))
        assert np.allclose(a.limits*.4,np.tile([5.28,5.28,11.2],4))
        assert abs(a.dt-.002)<1e-12 and a.foot_radius==.02
        checks+=['dimensions_and_mass','URDF_effort_limits','SDK_2ms_period','runtime_URDF_single_field_correction']
        names={'torso':'BASE_INER'}
        for leg,sdk in zip(LEGS,SDK_LEGS):
            names.update({leg+'_'+p:sdk+'_'+s for p,s in [('hip','HIP'),('thigh','THIGH'),('shank','SHANK'),('foot_body','FOOT')]})
        for internal,sdk in names.items():
            node=urdf.find(f"link[@name='{sdk}']/inertial");v={k:float(v) for k,v in node.find('inertia').attrib.items()}
            expected=np.array([[v['ixx'],v['ixy'],v['ixz']],[v['ixy'],v['iyy'],v['iyz']],[v['ixz'],v['iyz'],v['izz']]])
            i=a.model.body(internal).id;rot=np.zeros(9);mujoco.mju_quat2Mat(rot,a.model.body_iquat[i]);rot=rot.reshape(3,3)
            actual=rot@np.diag(a.model.body_inertia[i])@rot.T
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-15)
            assert abs(a.model.body_mass[i]-float(node.find('mass').get('value')))<1e-12
            origin=node.find('origin');com=np.fromstring(origin.get('xyz'),sep=' ') if origin is not None else np.zeros(3)
            np.testing.assert_allclose(a.model.body_ipos[i],com,atol=1e-14)
            eigen=np.sort(a.model.body_inertia[i]);assert eigen[0]>0 and eigen[0]+eigen[1]>=eigen[2]-1e-14
        checks.append('all_17_source_mass_COM_and_full_inertia_tensors')
        for info in manifest['joints']:
            joint=a.model.joint(info['simulation_joint'])
            np.testing.assert_allclose(joint.axis,info['axis'],atol=1e-14)
            np.testing.assert_allclose(joint.range,info['range_rad'],atol=1e-14)
            assert a.model.dof_damping[joint.dofadr[0]]==0 and a.model.dof_armature[joint.dofadr[0]]==0
        checks.append('all_12_joint_axes_ranges_and_source_damping')
        rng=np.random.default_rng(20261019)
        for _ in range(12):
            a.reset();q=a.neutral+rng.uniform(-.15,.15,12);a.data.qpos[a.qadr]=q;mujoco.mj_forward(a.model,a.data)
            for index,(leg,sdk) in enumerate(zip(LEGS,SDK_LEGS)):
                pos=a.data.qpos[:3].copy();rot=np.eye(3)
                for k,name in enumerate(('HipX','HipY','Knee','Ankle')):
                    j=urdf.find(f"joint[@name='{sdk}_{name}']");o=j.find('origin');assert o.get('rpy','0 0 0')=='0 0 0'
                    pos+=rot@np.fromstring(o.get('xyz','0 0 0'),sep=' ')
                    if k<3:rot=rot@rotation(np.fromstring(j.find('axis').get('xyz'),sep=' '),q[3*index+k])
                np.testing.assert_allclose(a.data.site_xpos[a.sites[index]],pos,atol=1e-12)
        checks.append('48_independent_URDF_foot_forward_kinematics')
        a.reset(seed=1);b.reset(seed=1)
        for _ in range(100):
            action=rng.uniform(-.4,.4,12);oa,ra,da,ta,ia=a.step(action);ob,rb,db,tb,ib=b.step(action)
            np.testing.assert_array_equal(a.data.qpos,b.data.qpos);np.testing.assert_array_equal(oa,ob)
            np.testing.assert_array_equal(a.data.ctrl,np.clip(action,-1,1)*a.cfg.action_scale*a.limits)
            assert np.isfinite(oa).all() and np.isfinite(ra) and np.isfinite(rb)
            if da or ta:a.reset(seed=1);b.reset(seed=1)
        checks+=['100_step_visual_headless_identical_dynamics','direct_neural_torque_only','finite_67D_observations']
        action=np.linspace(-3,3,12);a.reset();a.step(action)
        np.testing.assert_array_equal(a.data.ctrl,np.clip(action,-1,1)*.4*a.limits)
        checks.append('action_saturation_5.28_5.28_11.2_Nm')
        for invalid in (np.zeros(11),np.full(12,np.nan)):
            try:a.step(invalid)
            except ValueError:pass
            else:raise AssertionError('Invalid action accepted')
        checks.append('invalid_action_rejection')
        try:assert_policy_compatible({'robot_model':'mini_lite_sdk_v1','physics_dt':.002,'frame_skip':1})
        except ValueError:pass
        else:raise AssertionError('Old-model policy accepted')
        checks.append('old_policy_model_id_rejection')
        old=Path('runs/mini_lite/final/policy.zip')
        if old.exists():
            try:NumpyPolicy(old)
            except ValueError:pass
            else:raise AssertionError('Old archive accepted')
            diagnostic=NumpyPolicy(old,legacy_diagnostic=True)
            try:diagnostic.save(Path('results/forbidden.zip'))
            except ValueError:pass
            else:raise AssertionError('Diagnostic model promoted')
            checks.append('old_archive_rejected_and_diagnostic_save_blocked')
        try:QuadrupedEnv(Config(reference_reward=True))
        except ValueError:pass
        else:raise AssertionError('Old reference motion used')
        checks.append('unvalidated_old_reference_motion_rejected')
        return {'status':'PASS','model_id':MODEL_ID,'checks_passed':len(checks),'checks':checks,
                'mass_kg':float(a.model.body_mass.sum()),'torso_mass_kg':5.6352,'observations':67,'actions':12,
                'control_period_s':a.dt,'joint_effort_Nm':[13.2,13.2,28],'actual_action_limits_Nm':[5.28,5.28,11.2],
                'path_success':'REQUIRES_INDEPENDENT_POLICY_ACCEPTANCE','mjcf_sha256':sha256(ASSETS/'quadruped.xml')}
    finally:a.close();b.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='results/mini_lite_urdf/interface_check.json')
    args=p.parse_args();result=run_checks();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

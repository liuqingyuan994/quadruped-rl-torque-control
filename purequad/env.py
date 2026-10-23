"""Pure learned joint torques. Reference motion is consulted only by training rewards."""
from dataclasses import replace,asdict
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from .config import Config
from .camera import CameraTracker
from .model import ASSETS,LEGS,assert_policy_compatible
from .paths import Path,wrap


class QuadrupedEnv(gym.Env):
    metadata={'render_modes':['rgb_array'],'render_fps':60}

    def __init__(self,cfg=None,path='straight',visuals=False):
        super().__init__();self.cfg=replace(cfg) if cfg else Config()
        assert_policy_compatible(asdict(self.cfg))
        self.path_name=path;self.path=Path(path);self.visuals=visuals
        xml=(ASSETS/'quadruped.xml').read_text(encoding='utf-8');assets={}
        if visuals:assets={'mini_lite_meshes/'+p.name:p.read_bytes() for p in (ASSETS/'mini_lite_meshes').glob('*.stl')}
        else:
            tree=ET.fromstring(xml)
            for parent in tree.iter():
                for child in list(parent):
                    if child.tag=='mesh':parent.remove(child)
                    elif child.tag=='geom' and child.get('type')=='mesh':
                        if any(child.get(k)!='0' for k in ('mass','contype','conaffinity')):
                            raise ValueError('Only massless noncolliding visuals may be removed')
                        parent.remove(child)
            xml=ET.tostring(tree,encoding='unicode')
        self.model=mujoco.MjModel.from_xml_string(xml,assets=assets);self.model.opt.timestep=self.cfg.physics_dt
        self.data=mujoco.MjData(self.model);self.torso=self.model.body('torso').id
        self.qadr=np.array([self.model.joint(leg+'_'+j).qposadr[0] for leg in LEGS for j in ['abduction','hip_flexion','knee']])
        self.vadr=np.array([self.model.joint(leg+'_'+j).dofadr[0] for leg in LEGS for j in ['abduction','hip_flexion','knee']])
        self.ranges=np.array([self.model.joint(leg+'_'+j).range for leg in LEGS for j in ['abduction','hip_flexion','knee']])
        self.sites=[self.model.site(leg+'_foot').id for leg in LEGS]
        self.feet_ids=[self.model.geom(leg+'_foot_geom').id for leg in LEGS]
        self.limits=self.model.actuator_ctrlrange[:,1].copy();self.neutral=np.tile([0.,-.65,1.3],4)
        self.observation_space=spaces.Box(-10,10,(67,),dtype=np.float32)
        self.action_space=spaces.Box(-1,1,(12,),dtype=np.float32)
        self.action=np.zeros(12);self.renderer=None;self.camera_tracker=CameraTracker()
        self.appearance=None  # CAD meshes follow the actual body frames directly.
        self.foot_radius=float(self.model.geom_size[self.feet_ids[0],0])
        self.reference=None
        if self.cfg.reference_reward:
            raise ValueError('旧模型参考动作未针对新 URDF 关节轴验证。请关闭 --reference，使用新模型奖励训练。')

    @property
    def dt(self):return self.cfg.physics_dt*self.cfg.frame_skip

    def sample(self):
        R=self.data.xmat[self.torso].reshape(3,3)
        roll=np.arctan2(R[2,1],R[2,2]);pitch=np.arcsin(np.clip(-R[2,0],-1,1));yaw=np.arctan2(R[1,0],R[0,0])
        return R,np.array([roll,pitch,yaw]),self.data.qpos[:3].copy()

    def contacts(self):
        flags=np.zeros(4,dtype=bool);bad=0
        for c in self.data.contact[:self.data.ncon]:
            g1,g2=int(c.geom1),int(c.geom2)
            a,b=self.model.geom_bodyid[g1]==0,self.model.geom_bodyid[g2]==0
            if a!=b:
                robot=g2 if a else g1
                if robot in self.feet_ids:flags[self.feet_ids.index(robot)]=True
                else:bad+=1
        return flags,bad

    def observation(self,sample=None,track=None,contacts=None):
        R,angles,pos=sample if sample is not None else self.sample()
        progress,cross,tangent=track if track is not None else self.path.project(pos[:2])
        c,s=np.cos(angles[2]),np.sin(angles[2]);rot=np.array([[c,s],[-s,c]])
        future=np.concatenate([rot@(self.path.at(progress+d)-pos[:2]) for d in [.2,.6,1.]])
        head=wrap(tangent-angles[2]);phase=2*np.pi*max(0,self.data.time-self.cfg.clock_delay)/self.cfg.period
        gate=np.clip((self.data.time-self.cfg.clock_delay)/.15,0,1)
        clock=gate*np.array([f(k*phase) for k in [1,2,3] for f in [np.sin,np.cos]])
        obs=np.r_[R.T@np.array([0,0,-1]),R.T@self.data.qvel[:3],self.data.qvel[3:6],
            self.data.qpos[self.qadr]-self.neutral,self.data.qvel[self.vadr]/10,self.action,
            self.contacts()[0] if contacts is None else contacts,future,cross,np.sin(head),np.cos(head),
            (pos[2]-self.cfg.height)*10,0 if self.cfg.stage=='stand' else self.cfg.speed,clock,1.]
        return np.clip(np.nan_to_num(obs),-10,10).astype(np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);options=options or {}
        mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        if 'start_progress' in options:
            start=options['start_progress'];self.data.qpos[:2]=self.path.at(start)
            yaw=self.path.project(self.path.at(start+.001))[2]+options.get('yaw_offset',0)
            self.data.qpos[3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
        self.data.qpos[1]+=options.get('y_offset',0)
        noise=options.get('reset_noise',self.cfg.reset_noise)
        if noise:self.data.qpos[self.qadr]+=self.np_random.uniform(-noise,noise,12)
        mujoco.mj_forward(self.model,self.data);self.action[:]=0;self.steps=0
        self.last_progress=self.path.project(self.data.qpos[:2])[0]
        self.velocity_average=np.zeros(3)
        self.camera_tracker.reset(self.data.qpos[:3],self.data.time)
        return self.observation(),{'reason':'reset'}

    def reward_reference(self):
        ref=self.reference
        if self.data.time<=self.cfg.clock_delay:
            return self.neutral,np.zeros(12),np.full(4,self.foot_radius),np.ones(4)
        u=(self.data.time-self.cfg.clock_delay)/self.cfg.period%1*len(ref['q'])
        i=int(u)%len(ref['q']);j=(i+1)%len(ref['q']);a=u-int(u)
        return tuple((1-a)*ref[k][i]+a*ref[k][j] for k in ['q','dq','feet_z','contacts'])

    def step(self,action):
        a=np.asarray(action,dtype=float)
        if a.shape!=(12,) or not np.isfinite(a).all():raise ValueError('Policy must supply 12 finite normalized torques')
        previous=self.action.copy();self.action=np.clip(a,-1,1)*self.cfg.action_scale
        previous_dq=self.data.qvel[self.vadr].copy()
        torque=self.action*self.limits
        # 唯一电机控制写入：来自神经策略的完整关节力矩。
        self.data.ctrl[:]=torque
        power=float(np.sum(np.abs(torque*self.data.qvel[self.vadr])))
        mujoco.mj_step(self.model,self.data,nstep=self.cfg.frame_skip);mujoco.mj_forward(self.model,self.data);self.steps+=1
        sample=self.sample();R,angles,pos=sample;track=self.path.project(pos[:2]);progress,cross,tangent=track
        flags,bad=self.contacts();velocity=R.T@self.data.qvel[:3]
        self.velocity_average+=min(1.,self.dt/.15)*(velocity-self.velocity_average)
        margin=float(np.min(np.minimum(self.data.qpos[self.qadr]-self.ranges[:,0],self.ranges[:,1]-self.data.qpos[self.qadr])))
        finite=bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        fallen=pos[2]<self.cfg.min_height or pos[2]>.6 or max(abs(angles[0]),abs(angles[1]))>.7
        limited=margin<self.cfg.joint_margin;off_path=abs(cross)>1.
        success=self.cfg.stage=='path' and progress>=self.path.total-.1 and np.linalg.norm(pos[:2]-self.path.points[-1])<.2
        overspeed=bool(np.max(np.abs(self.data.qvel[self.vadr]))>self.cfg.joint_speed_limit_rad_s)
        failed=not finite or fallen or limited or off_path or overspeed
        terminated=bool(failed or success);truncated=self.steps*self.dt>=self.cfg.seconds and not terminated
        speed=0 if self.cfg.stage=='stand' or self.data.time<self.cfg.clock_delay else self.cfg.speed
        moving=1. if self.cfg.stage=='stand' else .1+.9*float(np.clip(velocity[0]/max(self.cfg.speed,.01),0,1))
        head=wrap(tangent-angles[2]);delta=progress-self.last_progress;self.last_progress=progress
        terms={'alive':self.dt,'upright':6*np.exp(-float(np.sum(angles[:2]**2))/.02)*self.dt*moving,
            'height':4*np.exp(-((pos[2]-self.cfg.height)/.02)**2)*self.dt*moving,
            'velocity':(2 if self.cfg.stage=='stand' else 10)*np.exp(-((velocity[0]-speed)**2+velocity[1]**2)/.04**2)*self.dt,
            'body_contact':-40*bad*self.dt,'joint_velocity':-.0002*float(np.sum(self.data.qvel[self.vadr]**2))*self.dt,
            'energy':-.00005*power*self.dt,'torque':-.02*float(np.mean(self.action**2))*self.dt,
            'smooth':-.02*float(np.mean((self.action-previous)**2)),
            'heading':-8*head**2*self.dt if self.cfg.stage!='stand' else 0,
            'cross_track':-120*cross**2*self.dt if self.cfg.stage=='path' else 0,
            'progress':20*delta if self.cfg.stage!='stand' else 0,
            'terminal':-self.cfg.fall_penalty if failed else 100 if success else 0}
        if self.cfg.reward_version>=2 and self.cfg.stage!='stand':
            error=(self.velocity_average[0]-speed)**2+self.velocity_average[1]**2
            terms['velocity']=10*np.exp(-error/.04**2)*self.dt
            terms['speed_error']=-150*error*self.dt
            terms['progress']=2*delta
            terms['cross_track']=-120*cross**2*self.dt
            terms['vertical_velocity']=-10*velocity[2]**2*self.dt
        if self.cfg.reward_version>=3 and self.cfg.stage!='stand':
            terms['upright']=8*np.exp(-float(np.sum(angles[:2]**2))/.002)*self.dt*moving
            terms['height']=5*np.exp(-((pos[2]-self.cfg.height)/.008)**2)*self.dt*moving
            terms['vertical_quiet']=2*np.exp(-(velocity[2]/.015)**2)*self.dt
        q=self.data.qpos[self.qadr];dq=self.data.qvel[self.vadr]
        ddq=(dq-previous_dq)/self.dt
        if self.cfg.reward_version>=4 and self.cfg.stage!='stand':
            # Training penalties only: the neural action remains the sole torque source.
            terms['joint_acceleration']=-.0025*float(np.mean(ddq**2))*self.dt
            terms['torque_rate']=-.1*float(np.mean(((self.action-previous)/self.dt)**2))*self.dt
            terms['body_rotation_rate']=-2*float(np.sum(self.data.qvel[3:5]**2))*self.dt
            terms['height']=6*np.exp(-((pos[2]-self.cfg.height)/.006)**2)*self.dt*moving
            terms['vertical_velocity']=-40*velocity[2]**2*self.dt
        if self.cfg.reward_version>=5 and self.cfg.stage!='stand':
            # Retuned training penalties for the supplied zero-armature URDF.
            # No filtering, torque source, physical parameter or gate changes.
            terms['joint_acceleration']=-1e-6*float(np.mean(ddq**2))*self.dt
            terms['torque_rate']=-1e-4*float(np.mean(((self.action-previous)/self.dt)**2))*self.dt

        if self.reference is not None and self.cfg.stage!='stand':
            qr,dqr,fr,cr=self.reward_reference()
            foot_z=self.data.site_xpos[self.sites,2]
            terms.update(motion_pose=4*np.exp(-float(np.mean((q-qr)**2))/.005)*self.dt,
                motion_velocity=2*np.exp(-float(np.mean((dq-dqr)**2))/.5)*self.dt,
                foot_clearance=8*np.exp(-float(np.mean((foot_z-fr)**2))/.006**2)*self.dt,
                contact_timing=3*float(1-np.mean(np.abs(flags.astype(float)-cr)))*self.dt)
        else:terms['pose']=2*np.exp(-float(np.mean((q-self.neutral)**2))/.02)*self.dt
        reason='nonfinite' if not finite else 'fallen' if fallen else 'joint_limit' if limited else 'joint_speed_limit' if overspeed else 'off_path' if off_path else 'success' if success else 'timeout' if truncated else 'running'
        info={'reason':reason,'is_success':reason=='success','time_s':float(self.data.time),'position':pos.tolist(),
            'rpy':angles.tolist(),'progress_m':progress,'cross_track_m':cross,'heading_error_rad':head,
            'velocity_m_s':velocity.tolist(),'joint_margin_rad':margin,'contacts':flags.tolist(),
            'nonfoot_contacts':bad,'feet_z':self.data.site_xpos[self.sites,2].tolist(),'torque_Nm':torque.tolist(),
            'joint_position_rad':q.tolist(),'joint_velocity_rad_s':dq.tolist(),'joint_acceleration_rad_s2':ddq.tolist(),
            'reward_terms':{k:float(v) for k,v in terms.items()},'control_architecture':'learned_joint_torques'}
        return self.observation(sample,track,flags),float(np.nan_to_num(sum(terms.values()),nan=-50)),terminated,truncated,info

    def render(self):
        if not self.visuals:raise RuntimeError('Create environment with visuals=True')
        if self.appearance is not None:self.appearance.update(self.data)
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,480,640)
        cam=mujoco.MjvCamera();cam.lookat[:]=self.camera_tracker.update(self.data.qpos[:3],self.data.time);cam.distance=1.5;cam.azimuth=135;cam.elevation=-25
        self.renderer.update_scene(self.data,camera=cam)
        self.draw_path(self.renderer.scene)
        return self.renderer.render().copy()

    def draw_path(self,scene):
        for x,y in self.path.points[::5]:
            if scene.ngeom>=scene.maxgeom:break
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3,.016),np.array([x,y,.015]),np.eye(3).ravel(),np.array([.1,.9,.45,1.]));scene.ngeom+=1
        if scene.ngeom<scene.maxgeom:
            x,y=self.path.points[-1]
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3,.045),np.array([x,y,.045]),np.eye(3).ravel(),np.array([1.,.15,.1,1.]));scene.ngeom+=1

    def close(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None

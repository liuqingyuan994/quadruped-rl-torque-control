"""Display-only tracking; never alters simulator state or policy inputs."""
import numpy as np


class CameraTracker:
    def __init__(self,mode='stable',time_constant=.20):
        if mode not in ('stable','raw'):raise ValueError('Camera mode must be stable or raw')
        self.mode=mode;self.time_constant=time_constant;self.target=None;self.last_time=None

    def reset(self,position,time=0.):
        self.target=np.asarray(position,dtype=float).copy();self.last_time=float(time)
        return self.target.copy()

    def update(self,position,time):
        position=np.asarray(position,dtype=float);time=float(time)
        if self.target is None or time<self.last_time:return self.reset(position,time)
        if self.mode=='raw':self.target[:]=position
        else:
            dt=max(0.,time-self.last_time)
            alpha=-np.expm1(-dt/self.time_constant)
            self.target[:2]+=alpha*(position[:2]-self.target[:2])
            # Keep the original world height, so gait bounce cannot move the floor.
        self.last_time=time
        return self.target.copy()

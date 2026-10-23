"""几何路径与误差测量，仅提供观察和奖励，不计算控制指令。"""
import numpy as np


def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


class Path:
    def __init__(self, kind="sine", length=5.0):
        x = np.linspace(0, length, 251)
        if kind == "straight":
            y = np.zeros_like(x)
        elif kind == "sine":
            y = 0.45 * (1-np.cos(2*np.pi*x/length))
        elif kind == "turn":
            x = np.concatenate([np.linspace(0,2,101), 2+np.sin(np.linspace(0,np.pi/2,101)), np.full(51,3.0)])
            y = np.concatenate([np.zeros(101), 1-np.cos(np.linspace(0,np.pi/2,101)),np.linspace(1,2,51)])
        else:
            raise ValueError(f"Unknown path {kind!r}: straight/sine/turn")
        self.points = np.column_stack((x,y))
        self.delta = np.diff(self.points,axis=0)
        self.lengths = np.linalg.norm(self.delta,axis=1)
        self.cumulative = np.r_[0,np.cumsum(self.lengths)]
        self.total = self.cumulative[-1]

    def at(self, progress):
        s = np.clip(progress,0,self.total)
        i = min(np.searchsorted(self.cumulative,s,side="right")-1,len(self.delta)-1)
        return self.points[i] + (s-self.cumulative[i])/max(self.lengths[i],1e-9)*self.delta[i]

    def project(self, xy):
        rel = xy-self.points[:-1]
        u = np.clip(np.sum(rel*self.delta,axis=1)/np.maximum(self.lengths**2,1e-12),0,1)
        projections = self.points[:-1]+u[:,None]*self.delta
        i = np.argmin(np.sum((xy-projections)**2,axis=1))
        tangent = self.delta[i]/max(self.lengths[i],1e-9)
        offset = xy-projections[i]
        cross = tangent[0]*offset[1]-tangent[1]*offset[0]
        return float(self.cumulative[i]+u[i]*self.lengths[i]),float(cross),float(np.arctan2(tangent[1],tangent[0]))


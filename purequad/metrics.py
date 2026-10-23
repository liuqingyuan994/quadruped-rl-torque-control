"""Joint vibration measurements; no optional SciPy dependency."""
import numpy as np


def high_frequency_rms(q,dt,cutoff=10.):
    q=np.asarray(q);n=min(1024,len(q))
    if n<8:return np.zeros(q.shape[1])
    window=np.hanning(n);spectra=[]
    for start in range(0,len(q)-n+1,max(1,n//2)):
        block=q[start:start+n];block=block-block.mean(axis=0)
        power=np.abs(np.fft.rfft(block*window[:,None],axis=0))**2/(np.sum(window**2)/dt)
        power[1:-1 if n%2==0 else None]*=2
        spectra.append(power)
    frequencies=np.fft.rfftfreq(n,dt)
    psd=np.mean(spectra,axis=0)
    return np.sqrt(np.sum(psd[frequencies>=cutoff],axis=0)/(n*dt))

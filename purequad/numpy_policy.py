"""Deterministic neural inference and weight editing without loading PyTorch DLLs.

Reads this project's SB3 float32 state dictionaries using an allowlisted unpickler.
It implements the existing Tanh MLP, state skip and path-clock interactions.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import collections
import io
import json
import pickle
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from .model import assert_policy_compatible


class Tensor:
    def __init__(self,storage,offset,size,stride,*unused):
        self.storage=storage;self.offset=offset;self.size=size;self.stride=stride


class Reader(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
        if module=='torch' and name=='FloatStorage':return np.dtype('<f4')
        if module=='torch._utils' and name=='_rebuild_tensor_v2':return Tensor
        raise ValueError(f'Unexpected state-dictionary class: {module}.{name}')

    def persistent_load(self,identifier):
        kind,dtype,key,device,count=identifier
        if kind!='storage' or dtype!=np.dtype('<f4'):raise ValueError('Requires float32 weights')
        return key,dtype,count


class NumpyPolicy:
    def __init__(self,filename,*,legacy_diagnostic=False):
        self.filename=str(filename)
        with ZipFile(filename) as archive:
            self.metadata=json.loads(archive.read('data'));self.policy_bytes=archive.read('policy.pth')
        self.pure_config=self.metadata['pure_config'].copy()
        self.legacy_diagnostic=legacy_diagnostic
        if legacy_diagnostic:
            if self.pure_config.get('robot_model')!='mini_lite_sdk_v1':
                raise ValueError('Diagnostic transfer accepts only the previous Mini Lite model')
            from .model import MODEL_ID
            assert_policy_compatible(dict(self.pure_config,robot_model=MODEL_ID))
        else:
            assert_policy_compatible(self.pure_config)
        with ZipFile(io.BytesIO(self.policy_bytes)) as archive:
            pkl=next(n for n in archive.namelist() if n.endswith('/data.pkl'))
            self.prefix=pkl[:-len('data.pkl')];self.tensors=Reader(io.BytesIO(archive.read(pkl))).load()
            self.weights={}
            for name,tensor in self.tensors.items():
                key,dtype,count=tensor.storage
                flat=np.frombuffer(archive.read(self.prefix+'data/'+key),dtype=dtype)
                if len(flat)!=count:raise ValueError('Invalid storage length')
                view=np.lib.stride_tricks.as_strided(flat[tensor.offset:],shape=tensor.size,
                    strides=tuple(s*dtype.itemsize for s in tensor.stride))
                self.weights[name]=view.copy()
        if self.weights['action_net.weight'].shape not in ((12,131),(12,143)):
            raise ValueError('Requires this project\'s 67-observation torque policy')

    def predict(self,obs,deterministic=True):
        if not deterministic:raise ValueError('NumPy replay uses deterministic inference only')
        x=np.asarray(obs,dtype=np.float32);w=self.weights
        latent=np.tanh(w['mlp_extractor.policy_net.0.weight']@x+w['mlp_extractor.policy_net.0.bias'])
        latent=np.tanh(w['mlp_extractor.policy_net.2.weight']@latent+w['mlp_extractor.policy_net.2.bias'])
        features=np.concatenate((latent,x))
        if w['action_net.weight'].shape[1]==143:
            features=np.concatenate((features,np.outer(x[[55,56]],x[60:66]).ravel()))
        return np.clip(w['action_net.weight']@features+w['action_net.bias'],-1,1),None

    def save(self,filename,configuration=None,provenance=None):
        if self.legacy_diagnostic:
            raise ValueError("Diagnostic legacy weights cannot be saved as a current-model policy")
        replacements={}
        for name,tensor in self.tensors.items():
            key,dtype,count=tensor.storage
            array=np.asarray(self.weights[name],dtype=dtype)
            if tensor.offset or np.prod(tensor.size)!=count or array.shape!=tensor.size:
                raise ValueError('Only whole contiguous weight storages are supported for editing')
            replacements[self.prefix+'data/'+key]=array.tobytes(order='C')
        storage=io.BytesIO()
        with ZipFile(io.BytesIO(self.policy_bytes)) as original,ZipFile(storage,'w',ZIP_DEFLATED) as output:
            for entry in original.infolist():output.writestr(entry,replacements.get(entry.filename,original.read(entry.filename)))
        metadata=self.metadata.copy()
        if configuration is not None:metadata['pure_config']=configuration
        if provenance is not None:metadata['smoothing_search']=provenance
        destination=Path(filename);destination.parent.mkdir(parents=True,exist_ok=True)
        with ZipFile(self.filename) as original,ZipFile(destination,'w',ZIP_DEFLATED) as output:
            for entry in original.infolist():
                content=storage.getvalue() if entry.filename=='policy.pth' else json.dumps(metadata,indent=2).encode() if entry.filename=='data' else original.read(entry.filename)
                output.writestr(entry,content)

"""Learned neural torque policy. State/clock skip weights are policy parameters."""
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor


class StateExtractor(MlpExtractor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.latent_dim_pi+=args[0]

    def forward_actor(self,features):
        return torch.cat((super().forward_actor(features),features),dim=1)


class TorquePolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor=StateExtractor(self.features_dim,net_arch=self.net_arch,
            activation_fn=self.activation_fn,device=self.device)


class PathExtractor(StateExtractor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.latent_dim_pi+=12

    def forward_actor(self,features):
        interactions=(features[:,[55,56],None]*features[:,None,60:66]).flatten(1)
        return torch.cat((super().forward_actor(features),interactions),dim=1)


class PathTorquePolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor=PathExtractor(self.features_dim,net_arch=self.net_arch,
            activation_fn=self.activation_fn,device=self.device)


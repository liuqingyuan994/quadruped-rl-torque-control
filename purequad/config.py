from dataclasses import dataclass


@dataclass
class Config:
    robot_model: str = 'mini_lite_urdf_20261019_v1'
    stage: str = 'stand'
    physics_dt: float = .002
    frame_skip: int = 1
    seconds: float = 5.
    speed: float = .13
    height: float = .31
    period: float = .70
    clock_delay: float = .6
    action_scale: float = .4
    reset_noise: float = 0.
    reference_reward: bool = False
    min_height: float = .235
    joint_margin: float = .12
    joint_speed_limit_rad_s: float = 20.
    foot_clearance_min_m: float = .009
    fall_penalty: float = 50.
    reward_version: int = 1
    policy_variant: str = 'state'


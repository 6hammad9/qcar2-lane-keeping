"""Loads lane_params.yaml and provides config access."""
import yaml
from dataclasses import dataclass
from typing import Tuple


@dataclass
class LaneConfig:
    # Image
    img_width: int
    img_height: int
    strip_top_fraction: float
    
    # Segmentation
    hsv_low: Tuple[int, int, int]
    hsv_high: Tuple[int, int, int]
    morph_close_iterations: int
    
    # Detection
    corridor_px: int
    split_fraction: float
    lane_width_px: int
    
    # Control
    kp: float
    kd: float
    drive_speed: float
    max_error_to_drive: float
    no_detection_stop_frames: int
    
    # Topics
    topic_image: str
    topic_cmd_vel: str
    topic_scan: str
    
    # Debug
    show_debug: bool
    
    # Trajectory - Pure pursuit
    traj_file: str
    traj_lookahead_distance: float
    traj_min_lookahead: float
    traj_max_lookahead: float
    
    # Trajectory - Speed
    traj_max_speed: float
    traj_min_speed: float
    traj_curve_speed: float
    traj_curvature_threshold: float
    
    # Trajectory - Tolerances
    traj_goal_tolerance: float
    traj_start_tolerance: float
    traj_angle_tolerance: float
    
    # Trajectory - Control
    traj_kp_angular: float
    traj_max_angular: float
    
    # Trajectory - Reverse
    traj_reverse_angle_threshold: float
    
    # Obstacle avoidance
    obs_stop_dist: float
    obs_slow_dist: float
    obs_following_distance: float
    obs_front_angle_range: float
    obs_side_angle_min: float
    obs_side_angle_max: float
    
    # Overtaking
    overtake_offset: float
    overtake_straight_check_distance: float
    overtake_max_heading_change: float
    overtake_min_side_clearance: float
    overtake_prefer_left: bool


def load_config(yaml_path: str) -> LaneConfig:
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    traj = cfg.get('trajectory', {})
    obs = cfg.get('obstacle', {})
    ovt = cfg.get('overtake', {})
    
    return LaneConfig(
        # Image
        img_width=cfg['image']['width'],
        img_height=cfg['image']['height'],
        strip_top_fraction=cfg['image']['strip_top_fraction'],
        
        # Segmentation
        hsv_low=tuple(cfg['segmentation']['hsv_low']),
        hsv_high=tuple(cfg['segmentation']['hsv_high']),
        morph_close_iterations=cfg['segmentation']['morph_close_iterations'],
        
        # Detection
        corridor_px=cfg['detection']['corridor_px'],
        split_fraction=cfg['detection']['split_fraction'],
        lane_width_px=cfg['detection']['lane_width_px'],
        
        # Control
        kp=cfg['control']['kp'],
        kd=cfg['control']['kd'],
        drive_speed=cfg['control']['drive_speed'],
        max_error_to_drive=cfg['control']['max_error_to_drive'],
        no_detection_stop_frames=cfg['control']['no_detection_stop_frames'],
        
        # Topics
        topic_image=cfg['topics']['image'],
        topic_cmd_vel=cfg['topics']['cmd_vel'],
        topic_scan=cfg['topics'].get('scan', '/rosbot3/scan'),
        
        # Debug
        show_debug=cfg['debug']['show_display'],
        
        # Trajectory - Pure pursuit
        traj_file=traj.get('file', ''),
        traj_lookahead_distance=traj.get('lookahead_distance', 0.5),
        traj_min_lookahead=traj.get('min_lookahead', 0.3),
        traj_max_lookahead=traj.get('max_lookahead', 1.0),
        
        # Trajectory - Speed
        traj_max_speed=traj.get('max_speed', 0.20),
        traj_min_speed=traj.get('min_speed', 0.05),
        traj_curve_speed=traj.get('curve_speed', 0.10),
        traj_curvature_threshold=traj.get('curvature_threshold', 0.5),
        
        # Trajectory - Tolerances
        traj_goal_tolerance=traj.get('goal_tolerance', 0.10),
        traj_start_tolerance=traj.get('start_tolerance', 0.15),
        traj_angle_tolerance=traj.get('angle_tolerance', 0.10),
        
        # Trajectory - Control
        traj_kp_angular=traj.get('kp_angular', 1.5),
        traj_max_angular=traj.get('max_angular', 1.0),
        
        # Trajectory - Reverse
        traj_reverse_angle_threshold=traj.get('reverse_angle_threshold', 120.0),
        
        # Obstacle avoidance
        obs_stop_dist=obs.get('stop_dist', 0.4),
        obs_slow_dist=obs.get('slow_dist', 0.8),
        obs_following_distance=obs.get('following_distance', 0.5),
        obs_front_angle_range=obs.get('front_angle_range', 30.0),
        obs_side_angle_min=obs.get('side_angle_min', 30.0),
        obs_side_angle_max=obs.get('side_angle_max', 90.0),
        
        # Overtaking
        overtake_offset=ovt.get('offset', 0.5),
        overtake_straight_check_distance=ovt.get('straight_check_distance', 2.0),
        overtake_max_heading_change=ovt.get('max_heading_change', 15.0),
        overtake_min_side_clearance=ovt.get('min_side_clearance', 0.6),
        overtake_prefer_left=ovt.get('prefer_left', True),
    )

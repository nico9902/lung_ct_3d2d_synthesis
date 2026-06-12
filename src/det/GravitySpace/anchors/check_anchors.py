import os
import sys
import numpy as np
import torch
import math
import types

# ----------------- #
# MOCK THE Missing 'net' structure
# ----------------- #
# Create mock modules to satisfy the imports in gravity_points_config
net = types.ModuleType('net')
sys.modules['net'] = net

net.utility = types.ModuleType('net.utility')
sys.modules['net.utility'] = net.utility

net.utility.msg = types.ModuleType('net.utility.msg')
sys.modules['net.utility.msg'] = net.utility.msg

# Mock functions
def mock_func(*args, **kwargs):
    pass

net.utility.msg.msg_config_complete = types.ModuleType('net.utility.msg_config_complete')
net.utility.msg.msg_config_complete.msg_config_complete = mock_func
sys.modules['net.utility.msg.msg_config_complete'] = net.utility.msg.msg_config_complete

net.utility.msg.msg_error = types.ModuleType('net.utility.msg_error')
net.utility.msg.msg_error.msg_error = mock_func
sys.modules['net.utility.msg.msg_error'] = net.utility.msg.msg_error

# Add the local directory to path for other local imports (anchors, local utility etc)
sys.path.append(os.getcwd())

# 1. First satisfy the local imports used by gravity_points_config
# from net.anchors.initial_config.initial_dice_config import initial_dice_config
# from net.anchors.initial_config.initial_grid_config import initial_grid_config
# from net.anchors.utility.shift import shift

# If 'net' package doesn't exist on disk, we can still link it to the src/det/GravitySpace path
gravity_space_path = os.path.join(os.getcwd(), 'src', 'det', 'GravitySpace')

from src.det.GravitySpace.anchors.initial_config.initial_grid_config import initial_grid_config
from src.det.GravitySpace.anchors.utility.shift import shift

# Define a modified gravity_points_config for this test (avoiding the complex path issues)
def gravity_points_config_test(step_val, image_shape):
    p = 5  # level 5 (of FPN)
    stride = 2 ** p  # stride = 32
    feature_map_shape = (image_shape + 2 ** p - 1) // (2 ** p)  # (12, 16) for (352, 480)
    
    # initial gravity points configuration grid
    gravity_points_initial_config = initial_grid_config(step=step_val,
                                                        image_shape=image_shape,
                                                        feature_map_shape=feature_map_shape)
    
    # shift gravity points over whole image
    shifted_gravity_points = shift(feature_map_shape=feature_map_shape,
                                   stride=stride,
                                   gravity_initial_config=gravity_points_initial_config)
    
    return shifted_gravity_points, gravity_points_initial_config, feature_map_shape

def main():
    image_shape = np.array([352, 480])  # (H, W)
    steps = [5, 6, 10, 15, 30]
    
    results = []
    
    # print headers
    print(f"\nGravity Points Analysis for Image Shape (352, 480):")
    print("-" * 75)
    print(f"{'Step':<10} | {'Pts/Pixel':<12} | {'FM Shape':<12} | {'Total Pts (A)':<15} | {'Grid (H x W)'}")
    print("-" * 75)
    
    for s in steps:
        gp, gp_init, fm_shape = gravity_points_config_test(s, image_shape)
        
        total_points = gp.shape[0]
        pts_per_pixel = gp_init.shape[0]
        
        # grid shape inside the reference window
        grid_h = math.ceil(352/12)
        grid_w = math.ceil(480/16)
        
        # points in grid
        h_pts = math.ceil(grid_h / s)
        w_pts = math.ceil(grid_w / s)

        print(f"{s:<10} | {pts_per_pixel:<12} | {fm_shape[0]}x{fm_shape[1]:<10} | {total_points:<15} | {h_pts}x{w_pts}")

if __name__ == "__main__":
    main()

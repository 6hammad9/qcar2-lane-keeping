import numpy as np

traj = np.load("/home/nvidia/ros2_ws/recorded_path_mpc_cart_16_06_02.npy")

start_idx = 1850
end_shift = 0.05 # 2.5 cm right

for i in range(start_idx, len(traj)):
    alpha = (i - start_idx) / (len(traj) - start_idx - 1)
    shift = alpha * end_shift

    yaw = traj[i, 2]

    traj[i, 0] += shift * np.sin(yaw)
    traj[i, 1] -= shift * np.cos(yaw)

np.save("recorded_path_mpc_cart_16_06_04_shifted.npy", traj)

print("Saved shifted trajectory")
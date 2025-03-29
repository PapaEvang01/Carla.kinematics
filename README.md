🚗 CARLA Kinematics

Predicting Future Positions of Vehicles in CARLA

📘 What is CARLA and Trajectory Prediction?

CARLA is an open-source simulator for autonomous driving research, offering realistic environments, sensors, and traffic scenarios.
 It enables safe testing of self-driving models in diverse urban settings with full control over vehicles, pedestrians, and weather.
Trajectory prediction refers to forecasting the future positions of moving agents (like vehicles) over time.
 In CARLA, it is essential for simulating and evaluating autonomous vehicle decision-making.
This project demonstrates simple kinematic-based trajectory prediction as a real-time baseline.

🚘 spawn20_kinematics.py — Kinematic Vehicle Prediction in CARLA

This script implements a real-time vehicle tracking and prediction pipeline.
 It spawns an ego vehicle and 20 NPC vehicles in the CARLA world, enables autopilot, and continuously monitors their state.
Using basic kinematic equations, it predicts future positions assuming constant speed and yaw (no acceleration or steering).
 Real-time logs of current and future states are printed to the terminal.

🧮 Kinematic Equations Used

This project uses constant-velocity motion modeling:
xf=x0+vx⋅tx_f
yf=y0+vy⋅ty_f 

Where:

x0,y0x_0, y_0 — Current position

xf,yfx_f, y_f — Predicted position after time tt

vx,vyv_x, v_y — Velocity components

tt — Prediction time window (e.g. 10 seconds)

Velocity is derived from vehicle speed and yaw:
vx=v⋅cos⁡(yaw),
vy=v⋅sin⁡(yaw)
This model acts as a baseline before applying learning-based or multi-agent trajectory predictors.

🎯 Advanced Kinematics Visualizations in CARLA
The project includes two upgraded scripts that visualize vehicle state and predictions directly inside the CARLA world.

📁 spawn_kinematics_draw_withoutspec.py

A visual extension of the base script with:

Real-time drawing of predicted trajectories
3D bounding boxes for all vehicles
Terminal logs for state + prediction
No camera movement — great for observing the whole map


📁 spawn_kinematics_draw_withspec.py

Builds on the above, adding:

A spectator camera that follows the ego vehicle from above
Ideal for visualizing local interactions around the ego vehicle
Maintains real-time drawings and logs






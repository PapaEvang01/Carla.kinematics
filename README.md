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


🧠 What is CRAT-Pred?

CRAT-Pred (Conditional Relational Attention Trajectory Prediction) is a deep neural network model that predicts future trajectories of multiple agents in traffic scenes.
It uses relational attention mechanisms to model interactions between nearby vehicles, making it ideal for dense, multi-agent environments like those in CARLA.

Unlike kinematic models, CRAT-Pred:

Learns social and spatial interactions from data

Produces multiple plausible futures (multi-modal predictions)

Selects the most likely trajectory based on learned patterns


🚀 CRAT-Pred in CARLA

These scripts extend the baseline kinematics version by integrating the CRAT-Pred deep learning model to perform real-time trajectory prediction in the CARLA simulator.

CRAT-Pred models multi-agent interactions and predicts multiple possible futures (multi-modal), making it much more realistic for autonomous driving scenarios.

📁 cratpred+carla_withspec.py
This is the fully integrated, ego-focused version that combines CRAT-Pred prediction and CARLA visualization with a top-down dynamic spectator camera.

Key Features:

Real-time connection to the CARLA simulator in asynchronous mode.

Spawns an ego vehicle (Tesla Model 3) and optional NPCs with autopilot.

Logs positions starting from 3 seconds to allow displacement history.

Computes normalized trajectories using rotation matrices based on movement direction.

Uses CRAT-Pred to predict future trajectories in real time.

Selects the best predicted trajectory using FDE (Final Displacement Error).

Transforms predictions back into world coordinates and visualizes them in CARLA.

Spectator camera follows the ego vehicle from above for intuitive tracking.

Automatically cleans up actors on exit.

✅ Best for ego-centric visualization of deep learning-based prediction.

📁 cratpred+carla_withoutspec.py
This version offers the same powerful CRAT-Pred integration as above, but without spectator movement, allowing a full static view of the simulation map.

Key Features:

Identical CRAT-Pred pipeline and vehicle spawning as the _withspec version.

Same logging, normalization, FDE-based mode selection, and visualization.

Static spectator view, showing the entire environment.

Great for observing interactions between multiple agents across the map.

✅ Best for global evaluation of agent predictions and behavior.




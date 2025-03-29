🚗 CARLA Kinematics

Predicting Future Positions of Vehicles in CARLA using Basic Kimetacis Equations and Deep Learning.


📘 What is CARLA and Trajectory Prediction?

CARLA is an open-source simulator for autonomous driving research, offering realistic environments, sensors, and traffic scenarios.
 It enables safe testing of self-driving models in diverse urban settings with full control over vehicles, pedestrians, and weather.
 
Trajectory prediction refers to forecasting the future positions of moving agents (like vehicles) over time.
 In CARLA, it is essential for simulating and evaluating autonomous vehicle decision-making.

 --> Basic Kimetacis Equations

🚘 spawn20_kinematics.py — Kinematic Vehicle Prediction in CARLA

This script implements a real-time vehicle tracking and prediction pipeline.
 It spawns an ego vehicle and 20 NPC vehicles in the CARLA world, enables autopilot, and continuously monitors their state.
Using basic kinematic equations, it predicts future positions assuming constant speed and yaw (no acceleration or steering).
 Real-time logs of current and future states are printed to the terminal.

🧮 Kinematic Equations Used

This project uses constant-velocity motion modeling:

xf=x0+vx⋅t, 
yf=y0+vy⋅t 

Where:

x0,y0 — Current position

xf,yf — Predicted position after time tt

vx,vy — Velocity components

t — Prediction time window (e.g. 10 seconds)

Velocity is derived from vehicle speed and yaw:

vx=v⋅cos⁡(yaw),
vy=v⋅sin⁡(yaw)

*This model acts as a baseline before applying learning-based or multi-agent trajectory predictors.

🎯 Advanced Kinematics Visualizations in CARLA

The project includes two upgraded scripts that visualize vehicle state and predictions directly inside the CARLA world.

📁 spawn_kinematics_draw_withoutspec.py

A visual extension of the base script with real-time drawing of predicted trajectories,3D bounding boxes for all vehicles,terminal logs for state + prediction,no camera movement.

📁 spawn_kinematics_draw_withspec.py

Builds on the above, adding a spectator camera that follows the ego vehicle from above

-->Deep Learning

🧠 What is CRAT-Pred?

CRAT-Pred (Conditional Relational Attention Trajectory Prediction) is a deep neural network model that predicts future trajectories of multiple agents in traffic scenes.
It uses relational attention mechanisms to model interactions between nearby vehicles, making it ideal for dense, multi-agent environments like those in CARLA.

Unlike kinematic models, CRAT-Pred learns social and spatial interactions from data,produces multiple plausible futures (multi-modal predictions),selects the most likely trajectory based on learned patterns

🚀 CRAT-Pred in CARLA

These scripts extend the baseline kinematics version by integrating the CRAT-Pred deep learning model to perform real-time trajectory prediction in the CARLA simulator.

CRAT-Pred models multi-agent interactions and predicts multiple possible futures (multi-modal), making it much more realistic for autonomous driving scenarios.

📁 cratpred+carla_withspec.py
This is the fully integrated, ego-focused version that combines CRAT-Pred prediction and CARLA visualization with a top-down dynamic spectator camera.

📁 cratpred+carla_withoutspec.py
This version offers the same powerful CRAT-Pred integration as above, but without spectator movement, allowing a full static view of the simulation map.





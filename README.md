# Carla.kinematics
Predicting Future Positions of Vehicles in CARLA 

What is CARLA and Trajectory Prediction?
CARLA is an open-source simulator for autonomous driving research, offering realistic environments, sensors, and traffic scenarios.
It allows safe testing of self-driving models in diverse urban settings with full control over vehicles, pedestrians, and weather.
Trajectory prediction refers to forecasting the future positions of moving agents (like vehicles) over time.
In CARLA, it's used to simulate and evaluate decision-making for autonomous vehicles.
This project demonstrates simple kinematic-based trajectory prediction as a baseline for real-time simulations.




Kinematics Equations for Future Position 

Using the basic kinematic equations, we predict the future position as:

xf=x0+vx⋅t
yf=y0+vy⋅t

Where:
xf,yf→ Future position
x0,y0​ → Current position

vx,vy→ Velocity components along X and Y
t → Prediction window (10 seconds)

The velocity components are computed using the vehicle’s speed and yaw (direction angle):

vx=v⋅cos⁡(yaw)
vy=v⋅sin⁡(yaw)

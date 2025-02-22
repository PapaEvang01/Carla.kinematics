# Carla.kinematics
Predicting Future Positions of Vehicles in CARLA using Kinematics


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

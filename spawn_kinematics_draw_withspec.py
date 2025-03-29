"""
Real-Time CARLA Vehicle Tracking & Visualization

Key Features:
- Connects to the CARLA simulator and sets the world to asynchronous mode.
- Spawns an ego vehicle (Tesla Model 3) along with 20 randomly selected NPC vehicles.
- Activates autopilot for all vehicles and continuously monitors their movement.
- Predicts and visualizes future trajectories based on velocity and orientation.
- Displays 3D bounding boxes around all vehicles for better spatial awareness.
- Keeps the spectator camera dynamically following the ego vehicle from above.
- Gracefully cleans up all actors when the simulation is manually stopped (Ctrl + C).

The spectator view provides a wide, top-down perspective of the ego.
"""


import carla
import random
import math
import time

# Constants
INITIAL_DELAY = 2.0  
UPDATE_INTERVAL = 1.0  
BBOX_LIFETIME = 5.0  # lifetime for bounding boxes


def connect_to_carla():
    """
    Connects to the CARLA simulator and ensures a valid connection.
    Returns:
        world (carla.World): The CARLA world object.
    """
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)  # Set a timeout for connection
        world = client.get_world()
        return world
    except Exception as e:
        print(f"Error connecting to CARLA: {e}")
        exit()

def ensure_async_mode(world):
    """
    Ensures that CARLA is running in asynchronous mode.
    Args:
        world (carla.World): The CARLA world object.
    """
    settings = world.get_settings()
    if settings.synchronous_mode:
        print("Disabling synchronous mode...")
        settings.synchronous_mode = False
        world.apply_settings(settings)

def get_spawn_points(world):
    """
    Retrieves and shuffles available spawn points in the CARLA world.
    Returns:
        list: A list of shuffled spawn points.
    """
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    if not spawn_points:
        print("No available spawn points. Exiting.")
        exit()
    return spawn_points

def spawn_ego_vehicle(world, spawn_points, blueprint_library):
    """
    Spawns the ego vehicle (Tesla Model 3) in the CARLA world.
    Returns:
        carla.Actor: The spawned ego vehicle actor.
    """
    ego_vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    if not spawn_points or not ego_vehicle_bp:
        print("Could not spawn Ego Vehicle. Exiting.")
        exit()
    
    ego_spawn_point = spawn_points.pop()
    ego_vehicle = world.try_spawn_actor(ego_vehicle_bp, ego_spawn_point)

    if ego_vehicle:
        ego_vehicle.set_autopilot(True)
        print(f"Ego Vehicle Spawned at {ego_spawn_point.location}")
        return ego_vehicle
    else:
        print("Failed to spawn Ego Vehicle.")
        exit()

def spawn_npc_vehicles(world, spawn_points, blueprint_library, num_vehicles=20):
    """
    Spawns NPC vehicles at available spawn points.
    Returns:
        list: A list of spawned NPC vehicle actors.
    """
    vehicles_list = []
    vehicle_blueprints = [bp for bp in blueprint_library.filter("vehicle.*") if bp.has_attribute('number_of_wheels')]

    for i in range(min(num_vehicles, len(spawn_points))):
        vehicle_bp = random.choice(vehicle_blueprints)
        spawn_point = spawn_points.pop()
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

        if vehicle:
            vehicle.set_autopilot(True)
            vehicles_list.append(vehicle)
            print(f"NPC Spawned: {vehicle.type_id} at {spawn_point.location}")
        else:
            print(f"Failed to spawn NPC vehicle {i+1}")

    return vehicles_list

def draw_bounding_boxes(world, vehicles):
    """
    Draws 3D bounding boxes around all vehicles 
    """
    for vehicle in vehicles:
        bounding_box = vehicle.bounding_box
        transform = vehicle.get_transform()
        vertices = bounding_box.get_world_vertices(transform)

        for i in range(4):
            world.debug.draw_line(vertices[i], vertices[(i+1) % 4], thickness=0.1, 
                                  color=carla.Color(255, 0, 0), life_time=BBOX_LIFETIME)
            world.debug.draw_line(vertices[i+4], vertices[((i+1) % 4) + 4], thickness=0.1, 
                                  color=carla.Color(255, 0, 0), life_time=BBOX_LIFETIME)
            world.debug.draw_line(vertices[i], vertices[i+4], thickness=0.1, 
                                  color=carla.Color(255, 0, 0), life_time=BBOX_LIFETIME)

def draw_predicted_trajectory(world, vehicles, prediction_time=5, step_size=0.5):
    """
    Draws the predicted trajectory for each vehicle based on velocity and yaw.
    Only shows waypoints for the future, not past positions.

    Args:
        world (carla.World): The CARLA world object.
        vehicles (list): List of vehicle actors.
        prediction_time (float): Time horizon for trajectory prediction (in seconds).
        step_size (float): Time step for drawing intermediate points.
    """
    for vehicle in vehicles:
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()

        # Extract vehicle kinematic data
        x_init = transform.location.x
        y_init = transform.location.y
        speed = math.sqrt(velocity.x**2 + velocity.y**2)
        yaw = transform.rotation.yaw  
        yaw_rad = math.radians(yaw)

        # Color: Ego vehicle = red, NPCs = blue
        color = carla.Color(255, 0, 0) if "ego" in vehicle.type_id else carla.Color(0, 0, 255)

        # Draw only **future trajectory points** for a few seconds
        for t in range(1, int(prediction_time / step_size) + 1):  # Start at 1 to skip current position
            time_step = t * step_size
            x_future = x_init + (speed * math.cos(yaw_rad) * time_step)
            y_future = y_init + (speed * math.sin(yaw_rad) * time_step)

            world.debug.draw_string(
                carla.Location(x_future, y_future, transform.location.z),
                "O",
                draw_shadow=False,
                color=color,
                life_time=2.0,  # Waypoints disappear after 2 seconds
                persistent_lines=False  # Don't persist waypoints
            )

def calculate_sides(hypotenuse, angle):
    """
    Calculates the two sides (x, y) of a right triangle given the hypotenuse and an angle.

    Args:
        hypotenuse (float): The hypotenuse length (distance behind the vehicle).
        angle (float): The yaw angle of the vehicle in degrees.

    Returns:
        tuple: The lengths of the two sides (delta_x, delta_y).
    """
    angle_radians = math.radians(angle)
    delta_x = hypotenuse * math.cos(angle_radians)
    delta_y = hypotenuse * math.sin(angle_radians)
    return delta_x, delta_y


def update_spectator_view(world, vehicle):
    """
    Updates the spectator camera to follow the Ego Vehicle from directly above.

    Args:
        world (carla.World): The CARLA world object.
        vehicle (carla.Actor): The Ego Vehicle actor.
    """
    spectator = world.get_spectator()
    vehicle_transform = vehicle.get_transform()
    
    follow_distance = -3   # Forward offset
    follow_height = 35     # Top-down height

    delta_x, delta_y = calculate_sides(follow_distance, vehicle_transform.rotation.yaw)

    spectator_transform = carla.Transform(
        vehicle_transform.location + carla.Location(x=-delta_x, y=-delta_y, z=follow_height),
        carla.Rotation(yaw=vehicle_transform.rotation.yaw, pitch=-90)
    )

    spectator.set_transform(spectator_transform)

def track_vehicles(vehicles_list, ego_vehicle, world):
    """
    Tracks vehicles, predicts future positions, and draws the future trajectory and the bounding boxes.
    Runs indefinitely until manually stopped.

    Args:
        vehicles_list (list): List of spawned NPC vehicles.
        ego_vehicle (carla.Actor): The ego vehicle actor.
        world (carla.World): The CARLA world object.
    """
    time_elapsed = INITIAL_DELAY
    try:
        while True:
            vehicle_data = []

            # Compute next prediction time
            prediction_time = time_elapsed + 1

            # Collect data for all vehicles (NPC + Ego)
            for vehicle in vehicles_list + [ego_vehicle]:
                transform = vehicle.get_transform()
                velocity = vehicle.get_velocity()

                x_init = transform.location.x
                y_init = transform.location.y
                speed = math.sqrt(velocity.x**2 + velocity.y**2)
                yaw = transform.rotation.yaw  

                vehicle_data.append((vehicle.id, "Ego" if vehicle == ego_vehicle else "NPC", x_init, y_init, speed, yaw))

            # Print vehicle data
            print(f"\nVehicle State at {time_elapsed:.0f}s (Prediction at {prediction_time}s)")
            print(f"{'ID':<5} {'Type':<10} {'X':<8} {'Y':<8} {'Speed':<8} {'Yaw':<8} {'X_future':<8} {'Y_future':<8}")
            print("=" * 90)

            for data in vehicle_data:
                vehicle_id, vehicle_type, x_init, y_init, speed, yaw = data
                yaw_rad = math.radians(yaw)  
                v_x = speed * math.cos(yaw_rad)
                v_y = speed * math.sin(yaw_rad)

                x_future = x_init + v_x * prediction_time
                y_future = y_init + v_y * prediction_time

                print(f"{vehicle_id:<5} {vehicle_type:<10} {x_init:<8.2f} {y_init:<8.2f} {speed:<8.2f} {yaw:<8.2f} {x_future:<8.2f} {y_future:<8.2f}")

            # Draw bounding boxes
            draw_bounding_boxes(world, vehicles_list + [ego_vehicle])

            # Draw the future predicted trajectory
            draw_predicted_trajectory(world, vehicles_list + [ego_vehicle], prediction_time=5, step_size=0.5)
		
            update_spectator_view(world, ego_vehicle)	

            print(f"Tracking vehicles at {time_elapsed:.0f}s")
            time.sleep(UPDATE_INTERVAL)
            time_elapsed += UPDATE_INTERVAL

    except KeyboardInterrupt:
        print("\nSimulation manually stopped.")
        cleanup_vehicles(vehicles_list, ego_vehicle)


def cleanup_vehicles(vehicles_list, ego_vehicle):
    """
    Removes all spawned vehicles from the simulation.
    Args:
        vehicles_list (list): List of spawned NPC vehicles.
        ego_vehicle (carla.Actor): The ego vehicle actor.
    """
    print("\nRemoving all spawned vehicles...")
    for vehicle in vehicles_list:
        vehicle.destroy()

    if ego_vehicle:
        ego_vehicle.destroy()
        print("Ego Vehicle removed.")

    print("All vehicles removed.")

def main():
    """
    Main function that initializes the CARLA world, spawns vehicles, and starts vehicle tracking.
    Runs indefinitely until interrupted.
    """
    # Connect to CARLA and ensure asynchronous mode
    world = connect_to_carla()
    ensure_async_mode(world)

    # Get spawn points and vehicle blueprints
    spawn_points = get_spawn_points(world)
    blueprint_library = world.get_blueprint_library()

    # Spawn ego vehicle and NPC vehicles
    ego_vehicle = spawn_ego_vehicle(world, spawn_points, blueprint_library)
    vehicles_list = spawn_npc_vehicles(world, spawn_points, blueprint_library)

    print("\nWaiting for vehicles to start moving...")
    time.sleep(INITIAL_DELAY)

    # Start tracking vehicle positions
    track_vehicles(vehicles_list, ego_vehicle, world)

if __name__ == "__main__":
    main()

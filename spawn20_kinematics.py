"""
Key Features:
- Connects to the CARLA simulator and ensures asynchronous mode is enabled.
- Spawns an ego vehicle (Tesla Model 3) at a random location.
- Spawns 20 NPC vehicles of random types and activates autopilot for each.
- Continuously tracks positions, speeds, and orientations of all vehicles.
- Predicts future positions in real-time based on current velocity and heading.
- Gracefully cleans up all vehicles upon manual termination (Ctrl + C).
"""


import carla
import random
import math
import time

# Constants
INITIAL_DELAY = 2.0  
UPDATE_INTERVAL = 1.0  

def connect_to_carla():
    """
    Connects to the CARLA simulator.
    Returns:
        world (carla.World): The world object representing the CARLA simulation.
    """
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)  
        world = client.get_world()
        return world
    except Exception as e:
        print(f"Error connecting to CARLA: {e}")
        exit()

def ensure_async_mode(world):
    """
    Ensures that CARLA is running in asynchronous mode.
    If synchronous mode is enabled, it disables it.
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
    Args:
        world (carla.World): The CARLA world object.
    Returns:
        list: A list of spawn points.
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
    Args:
        world (carla.World): The CARLA world object.
        spawn_points (list): List of available spawn points.
        blueprint_library (carla.BlueprintLibrary): The blueprint library for available vehicles.
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
    Args:
        world (carla.World): The CARLA world object.
        spawn_points (list): List of available spawn points.
        blueprint_library (carla.BlueprintLibrary): The blueprint library for available vehicles.
        num_vehicles (int): Number of NPC vehicles to spawn.
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

def track_vehicles(vehicles_list, ego_vehicle):
    """
    Continuously tracks and prints vehicle states with dynamically increasing future position prediction.
    Runs indefinitely until manually stopped.
    Args:
        vehicles_list (list): List of spawned NPC vehicles.
        ego_vehicle (carla.Actor): The ego vehicle actor.
    """
    time_elapsed = INITIAL_DELAY
    while True:  # Runs indefinitely
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

        # Print table header
        print(f"\nVehicle State at {time_elapsed:.0f}s (Prediction at {prediction_time}s)")
        print(f"{'ID':<5} {'Type':<10} {'X':<8} {'Y':<8} {'Speed':<8} {'Yaw':<8} {'X_future':<8} {'Y_future':<8}")
        print("=" * 90)

        # Compute and print future positions
        for data in vehicle_data:
            vehicle_id, vehicle_type, x_init, y_init, speed, yaw = data
            yaw_rad = math.radians(yaw)  
            v_x = speed * math.cos(yaw_rad)
            v_y = speed * math.sin(yaw_rad)

            x_future = x_init + v_x * prediction_time
            y_future = y_init + v_y * prediction_time

            print(f"{vehicle_id:<5} {vehicle_type:<10} {x_init:<8.2f} {y_init:<8.2f} {speed:<8.2f} {yaw:<8.2f} {x_future:<8.2f} {y_future:<8.2f}")

        time.sleep(UPDATE_INTERVAL)
        time_elapsed += UPDATE_INTERVAL

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

    try:
        # Start tracking vehicle positions
        track_vehicles(vehicles_list, ego_vehicle)
    except KeyboardInterrupt:
        # If stopped manually, clean up vehicles
        print("\nSimulation manually stopped.")
        cleanup_vehicles(vehicles_list, ego_vehicle)

if __name__ == "__main__":
    main()


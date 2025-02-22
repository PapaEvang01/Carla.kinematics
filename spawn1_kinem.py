    """
What This Script Does
 Connects to CARLA
Spawns one moving vehicle
Continuously tracks its position, speed, and direction
 Predicts future position dynamically
Runs indefinitely (stops when manually terminated)
Removes the vehicle on exit
    """
import carla
import math
import time

# Constants
INITIAL_DELAY = 2.0  
UPDATE_INTERVAL = 1.0  

def connect_to_carla():
    """
    Connects to the CARLA simulator and returns the world object.
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
    """
    settings = world.get_settings()
    if settings.synchronous_mode:
        print("Disabling synchronous mode...")
        settings.synchronous_mode = False
        world.apply_settings(settings)

def get_spawn_points(world):
    """
    Retrieves and returns spawn points from the CARLA map.
    """
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No available spawn points. Exiting.")
        exit()
    return spawn_points

def spawn_vehicle(world, spawn_points, blueprint_library):
    """
    Spawns a single vehicle in the CARLA world.
    """
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')  # You can change the model if needed
    spawn_point = spawn_points[0]  # Pick the first available spawn point

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    if vehicle:
        vehicle.set_autopilot(True)  # Enable autopilot to make it move
        print(f"Vehicle Spawned at {spawn_point.location}")
        return vehicle
    else:
        print("Failed to spawn the vehicle.")
        exit()

def track_vehicle(vehicle):
    """
    Continuously tracks the vehicle's position and predicts its future position.
    Runs indefinitely until manually stopped.
    """
    time_elapsed = INITIAL_DELAY
    while True:  # Runs indefinitely
        # Get vehicle transform (position) and velocity
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()

        # Extract current values
        x_init = transform.location.x
        y_init = transform.location.y
        speed = math.sqrt(velocity.x**2 + velocity.y**2)  # Compute speed from velocity vector
        yaw = transform.rotation.yaw  # Orientation in degrees

        # Compute future prediction
        prediction_time = time_elapsed + 1  # Increase prediction dynamically
        yaw_rad = math.radians(yaw)  
        v_x = speed * math.cos(yaw_rad)
        v_y = speed * math.sin(yaw_rad)

        x_future = x_init + v_x * prediction_time
        y_future = y_init + v_y * prediction_time

        # Print vehicle state and predicted future position
        print(f"\nTime Elapsed: {time_elapsed:.0f}s (Prediction at {prediction_time}s)")
        print("=" * 60)
        print(f"{'ID':<5} {'X':<10} {'Y':<10} {'Speed':<10} {'Yaw':<10} {'X_future':<10} {'Y_future':<10}")
        print(f"{vehicle.id:<5} {x_init:<10.2f} {y_init:<10.2f} {speed:<10.2f} {yaw:<10.2f} {x_future:<10.2f} {y_future:<10.2f}")
        print("=" * 60)

        time.sleep(UPDATE_INTERVAL)
        time_elapsed += UPDATE_INTERVAL

def cleanup_vehicle(vehicle):
    """
    Removes the vehicle from the simulation when the script stops.
    """
    print("\nRemoving the spawned vehicle...")
    vehicle.destroy()
    print("Vehicle removed.")

def main():
    """
    Main function to connect to CARLA, spawn one vehicle, and track its position.
    Runs indefinitely until manually stopped.
    """
    world = connect_to_carla()
    ensure_async_mode(world)
    spawn_points = get_spawn_points(world)
    blueprint_library = world.get_blueprint_library()

    vehicle = spawn_vehicle(world, spawn_points, blueprint_library)

    print("\nWaiting for the vehicle to start moving...")
    time.sleep(INITIAL_DELAY)

    try:
        track_vehicle(vehicle)
    except KeyboardInterrupt:
        print("\nSimulation manually stopped.")
        cleanup_vehicle(vehicle)

if __name__ == "__main__":
    main()


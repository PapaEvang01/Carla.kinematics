"""
Trajectory Prediction in CARLA using CRAT-Pred
------------------------------------------------
This script connects to the CARLA simulator and performs real-time trajectory prediction
for an ego vehicle and optionally spawned NPC vehicles using the CRAT-Pred deep learning model.

Key Features:
- Connects to CARLA in asynchronous mode.
- Spawns a Tesla Model 3 as the ego vehicle, along with random NPC vehicles.
- Logs vehicle positions over time starting from 3 seconds.
- Calculates displacements and normalizes trajectories using rotation matrices
  based on the vehicle's last movement direction.
- Feeds normalized data into CRAT-Pred to predict future positions.
- Selects the most likely prediction mode using Final Displacement Error (FDE).
- Inversely rotates predictions back into world space and visualizes them in CARLA.
- Keeps the spectator camera locked top-down on the ego vehicle for clarity.
- Automatically cleans up vehicles on simulation stop.

This setup enables end-to-end evaluation of multi-modal trajectory predictions within
a realistic driving simulation environment.
"""

import time
import random
import numpy as np
import torch
import carla
import sys
import os
import math

# Ensure Python can find the "model/" directory
sys.path.append("/home/user/PycharmProjects/crat-pred/model")

# Try importing the CRAT-Pred model
from crat_pred import CratPred  

vehicle_histories = {}  # Stores historical (x, y) positions for each vehicle
vehicle_origins = {}  # Stores the first recorded position of each vehicle
vehicle_positions_log = {}  # Stores all recorded positions per second (timestamp → positions)1
vehicle_centers_matrix = {}  # Stores the processed center positions of each vehicle for trajectory prediction

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

    Args:
        world (carla.World): The CARLA world instance where the vehicle will be spawned.
        spawn_points (list): A list of available spawn points for vehicle placement.
        blueprint_library (carla.BlueprintLibrary): CARLA's blueprint library for retrieving vehicle definitions.

    Returns:
        carla.Actor: The spawned ego vehicle instance if successful.

    """
    ego_vehicle_bp = blueprint_library.find('vehicle.tesla.model3')  # Get the Tesla Model 3 blueprint

    if not spawn_points or not ego_vehicle_bp:
        print("Could not spawn Ego Vehicle. Exiting.") 
        exit()  # Exit if there are no spawn points or blueprint issues

    ego_spawn_point = spawn_points.pop() 
    ego_vehicle = world.try_spawn_actor(ego_vehicle_bp, ego_spawn_point)  

    if ego_vehicle:
        ego_vehicle.set_autopilot(True)  # Enable autopilot mode 
        print(f"Ego Vehicle Spawned at {ego_spawn_point.location}")  
        return ego_vehicle  # Return the spawned vehicle instance
    else:
        print("Failed to spawn Ego Vehicle.") 
        exit()  # Exit

def spawn_vehicle(world, spawn_points, blueprint_library, vehicle_type):
    """
    Spawns a vehicle in the CARLA simulation at an available spawn point.

    Args:
        world (carla.World): The CARLA world instance where the vehicle will be spawned.
        spawn_points (list): A list of available spawn points for vehicle placement.
        blueprint_library (carla.BlueprintLibrary): CARLA's blueprint library for retrieving vehicle definitions.
        vehicle_type (str): The type of vehicle to spawn (e.g., "vehicle.tesla.model3", "vehicle.audi.etron").

    Returns:
        carla.Actor: The spawned vehicle instance if successful.
        None: If no valid spawn points are available or if spawning fails.

    """
    vehicle_bp = blueprint_library.find(vehicle_type) 

    if not spawn_points or not vehicle_bp:
        print(f"Could not spawn {vehicle_type}. Exiting.")  
        return None  

    spawn_point = spawn_points.pop()  
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point) 

    if vehicle:
        vehicle.set_autopilot(True)  
        vehicle_histories[vehicle.id] = []  # Initialize history tracking for this vehicle
        print(f"Vehicle {vehicle_type} Spawned at Location(x={spawn_point.location.x:.6f}, y={spawn_point.location.y:.6f}, z={spawn_point.location.z:.6f})")
        return vehicle  # Return the successfully spawned vehicle instance

    else:
        print(f"Failed to spawn {vehicle_type}. Retrying at a new location...") 

        if spawn_points:  
            return spawn_vehicle(world, spawn_points, blueprint_library, vehicle_type)  

        print("No valid spawn points left.")  
        return None 


def get_best_checkpoint(checkpoint_dir="/home/user/PycharmProjects/crat-pred/lightning_logs/version_51/checkpoints"):
    """
    Finds the best CRAT-Pred model checkpoint by selecting the one with the lowest fde_val.

    Args:
        checkpoint_dir (str): Path to the directory containing model checkpoint files.

    Returns:
        str: The path to the best checkpoint file.
    """
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]

    def extract_fde(filename):
        try:
            return float(filename.split("fde_val=")[-1].split("-")[0])
        except:
            return float("inf")

    best_checkpoint = min(checkpoints, key=extract_fde)
    return os.path.join(checkpoint_dir, best_checkpoint)


def load_cratpred_model():
    """
    Loads the CRAT-Pred model using the best available checkpoint.

    Returns:
        CratPred: The trained CRAT-Pred model in evaluation mode.
    """
    best_checkpoint = get_best_checkpoint()
    print(f"Loading best checkpoint: {best_checkpoint}")

    model = CratPred.load_from_checkpoint(best_checkpoint, strict=False)  # Load the model without strict version checks
    model.eval()  # Set model to evaluation mode
    return model


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
    Updates the spectator camera to **follow the Ego Vehicle from directly above** while showing a **wider view of the road**.

    Args:
        world (carla.World): The CARLA world object.
        vehicle (carla.Actor): The Ego Vehicle actor.
    """
    spectator = world.get_spectator()
    vehicle_transform = vehicle.get_transform()
    
    # Spectator camera positioning parameters
    follow_distance = -3  # Slightly in front
    follow_height = 35   # view of the road

    # Calculate slight backward offset
    delta_x, delta_y = calculate_sides(follow_distance, vehicle_transform.rotation.yaw)

    # Set the spectator 
    spectator_transform = carla.Transform(
        vehicle_transform.location + carla.Location(x=-delta_x, y=-delta_y, z=follow_height),
        carla.Rotation(yaw=vehicle_transform.rotation.yaw, pitch=-90)  # True top-down view
    )

    # Apply instant update without pauses
    spectator.set_transform(spectator_transform)


def calculate_rotation(vehicle_id):
    """
    Computes the rotation matrix that aligns the last movement vector
    of the vehicle's trajectory with the X-axis (standard CRAT-Pred normalization).

    Args:
        vehicle_id (int): The unique ID of the vehicle.

    Returns:
        torch.Tensor: A (1, 2, 2) rotation matrix.
    """

    # If we don't have enough position history for this vehicle (less than 2), return identity
    if vehicle_id not in vehicle_centers_matrix or vehicle_centers_matrix[vehicle_id].shape[1] < 2:
        print(f"[DEBUG] Vehicle {vehicle_id}: Not enough history for rotation. Returning identity.")
        return torch.eye(2, dtype=torch.float32).unsqueeze(0)

    # Extract the last two recorded positions of the vehicle
    positions = vehicle_centers_matrix[vehicle_id].squeeze(0).numpy()
    prev, curr = positions[-2], positions[-1]

    # Compute the movement vector (difference between last two positions)
    dx, dy = curr[0] - prev[0], curr[1] - prev[1]
    norm = np.linalg.norm([dx, dy])  # Get the magnitude of the movement vector

    # If the vehicle hasn’t moved (zero vector), return identity matrix
    if norm == 0:
        print(f"[DEBUG] Vehicle {vehicle_id}: No movement detected. Returning identity rotation.")
        return torch.eye(2, dtype=torch.float32).unsqueeze(0)

    # Compute the angle between the movement vector and the positive X-axis (in radians)
    angle = np.arctan2(dy, dx)
    angle_degrees = np.degrees(angle)  # For debugging

    # Construct a 2D rotation matrix to rotate the movement vector to align with the X-axis
    # We rotate by -angle to align the motion vector with the +X axis
    rotation = torch.tensor([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle),  np.cos(-angle)]
    ], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 2, 2)

    # DEBUG PRINTS 
    #print(f"[DEBUG] Vehicle {vehicle_id} rotation calculation:")
    #print(f"        Last movement vector: dx = {dx:.6f}, dy = {dy:.6f}")
    #print(f"        Angle to X-axis: {angle_degrees:.4f}°")
    #print(f"        Rotation Matrix:\n{rotation.squeeze(0).numpy()}")

    # Return the rotation matrix for normalization
    return rotation


def draw_predicted_trajectory(world, vehicles, prediction_time=5, step_size=0.5):
    """
    Draws the predicted waypoints for each vehicle based on CRAT-Pred predictions.

    - First **3 waypoints** are **green**.
    - Remaining waypoints are **blue**.
    - Displays the **first 10 predicted positions**.
    - Uses CARLA's debug draw utility.

    Args:
        world (carla.World): The CARLA world object.
        vehicles (list): List of CARLA vehicle actors whose trajectories are being visualized.
        prediction_time (int): Number of seconds to predict into the future.
        step_size (float): Time step between predicted waypoints.
    """
    max_waypoints = 10  # Display first 10 waypoints

    for vehicle in vehicles:
        vehicle_id = vehicle.id

        # Ensure we have predictions for this vehicle
        if f"predicted_{vehicle_id}" not in vehicle_positions_log:
            print(f"DEBUG: No predicted positions found for Vehicle {vehicle_id}")
            continue

        # Retrieve predicted trajectory positions
        predicted_positions = vehicle_positions_log[f"predicted_{vehicle_id}"]

        # Ensure there are predicted positions
        if not predicted_positions or len(predicted_positions) < 2:
            print(f"DEBUG: Not enough predicted positions for Vehicle {vehicle_id}")
            continue

        # **Select first 10 waypoints**
        waypoints_to_draw = predicted_positions[:max_waypoints]

        # Draw waypoints
        for idx, (waypoint_x, waypoint_y) in enumerate(waypoints_to_draw):
            # **First 3 waypoints = GREEN, the rest = BLUE**
            color = carla.Color(0, 255, 0) if idx < 3 else carla.Color(0, 0, 255)

            world.debug.draw_string(
                carla.Location(float(waypoint_x), float(waypoint_y), float(vehicle.get_transform().location.z)),
                "O",  # Waypoint symbol
                draw_shadow=False,
                color=color,
                life_time=1, 
                persistent_lines=False  # No connecting lines
            )

        #print(f"[INFO] Visualized {len(waypoints_to_draw)} waypoints for Vehicle {vehicle_id}")


def predict_future_cratpred(vehicle, model, timestamp):
    """
    Predicts future trajectories for a vehicle using the CRAT-Pred model with rotation normalization.

    Args:
        vehicle (carla.Actor): The CARLA vehicle instance being tracked.
        model (torch.nn.Module): The trained CRAT-Pred model.
        timestamp (int): The current simulation timestamp in seconds.

    Returns:
        list: A list of predicted (x, y) positions for the next timesteps.
    """
    print(f"\n[INFO] Processing Vehicle {vehicle.type_id} (ID: {vehicle.id}) at {timestamp}s")

    # ---  Get current position of the vehicle ---
    transform = vehicle.get_transform()
    #current_x, current_y = round(transform.location.x, 3), round(transform.location.y, 3)
    current_x, current_y = transform.location.x, transform.location.y 

    if vehicle.id not in vehicle_origins:
        vehicle_origins[vehicle.id] = [current_x, current_y]
        print(f"[DEBUG] Stored First Recorded Position for Vehicle {vehicle.id}: (X: {current_x}, Y: {current_y})")

    vehicle_histories.setdefault(vehicle.id, []).append([current_x, current_y])
    if len(vehicle_histories[vehicle.id]) > 21:
        vehicle_histories[vehicle.id].pop(0)

    numeric_keys = sorted([t for t in vehicle_positions_log.keys() if isinstance(t, int)])
    centers_list = [vehicle_positions_log[t][vehicle.id] for t in numeric_keys if vehicle.id in vehicle_positions_log[t]]

    if not centers_list:
        centers_list.append([current_x, current_y])

    centers_matrix = torch.tensor(centers_list, dtype=torch.float32).unsqueeze(0)
    vehicle_centers_matrix[vehicle.id] = centers_matrix

    centers_matrix_np = centers_matrix.squeeze(0).numpy()

    # ---  Calculate displacements from positions (first difference) ---
    if timestamp == 3:
        print("[DEBUG] Skipping prediction at t=3s to establish initial position.")
        return

    if timestamp == 4:
        first_displacement = np.array([[centers_matrix_np[-1, 0] - centers_matrix_np[-2, 0],
                                        centers_matrix_np[-1, 1] - centers_matrix_np[-2, 1]]])
        displacements = first_displacement
    elif centers_matrix_np.shape[0] > 1:
        displacements = np.diff(centers_matrix_np, axis=0)
    else:
        displacements = np.array([[0.0, 0.0]])
        print(f"[DEBUG] Not enough history for displacement at {timestamp}s. Using (0,0) for Vehicle {vehicle.id}.")

    displacements_tensor = torch.tensor(displacements, dtype=torch.float32).unsqueeze(0)
    ones_feature = torch.ones((displacements_tensor.shape[0], displacements_tensor.shape[1], 1), dtype=torch.float32)
    displacements_tensor = torch.cat((displacements_tensor, ones_feature), dim=-1)

    if 3 in vehicle_positions_log and vehicle.id in vehicle_positions_log[3]:
        origin_x, origin_y = vehicle_positions_log[3][vehicle.id]
    else:
        print(f"[ERROR] Missing t=3s position for Vehicle {vehicle.id}. Exiting prediction.")
        return None

    origin = torch.tensor([origin_x, origin_y], dtype=torch.float32).view(1, 2)
    centers = centers_matrix[:, -1, :].view(1, 2)

    # ---  Compute rotation matrix to align trajectory with the X-axis ---
    rotation_matrix = calculate_rotation(vehicle.id)
    if isinstance(rotation_matrix, np.ndarray):
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)
    elif rotation_matrix.dim() == 3:
        rotation_matrix = rotation_matrix.squeeze(0)

     # --- Apply rotation to all inputs (displacements, center, origin) ---
    displ_rotated = torch.matmul(displacements_tensor[:, :, :2], rotation_matrix.T)
    centers_rotated = torch.matmul(centers, rotation_matrix.T)
    origin_rotated = torch.matmul(origin, rotation_matrix.T)
    displ_rotated = torch.cat((displ_rotated, ones_feature), dim=-1)

    # --- Build CRAT-Pred batch input ---
    batch = {
        "displ": (displ_rotated,),
        "centers": (centers_rotated,),
        "rotation": rotation_matrix.unsqueeze(0),
        "origin": origin_rotated
    }

    # --- Run the prediction model ---
    with torch.no_grad():
        predictions = model(batch)

    if predictions is None:
        print(f"[ERROR] CRAT-Pred returned None for Vehicle {vehicle.id}. Skipping prediction.")
        return None

    predictions = predictions.squeeze(0).cpu().numpy()  # (num_modes, num_steps, 2)
    if predictions.ndim == 4 and predictions.shape[0] == 1:
    	predictions = predictions[0]  # squeeze batch
    num_modes = predictions.shape[0]

    if predictions.ndim != 3 or num_modes == 0:
        print(f"[ERROR] Invalid prediction shape: {predictions.shape}. Skipping prediction.")
        return None

    # ---  Evaluate FDE (Final Displacement Error) and pick the best mode ---
    real_last_x, real_last_y = centers.numpy().flatten()
    final_positions = predictions[:, -1, :]  # (num_modes, 2)
    fde_scores = np.linalg.norm(final_positions - [real_last_x, real_last_y], axis=1)
   # print(f"[DEBUG] FDE Scores: {fde_scores}")

    best_mode_index = int(np.argmin(fde_scores))
    best_fde_score = fde_scores[best_mode_index] if fde_scores.ndim == 1 else fde_scores.flatten()[best_mode_index]
    #print(f"[DEBUG] Selected best mode index: {best_mode_index} with FDE = {best_fde_score:.4f}")

    best_mode_displacements = predictions[best_mode_index]  # (num_steps, 2)
    #print(f"[DEBUG] Best mode displacements (raw):\n{best_mode_displacements}")

    # Inverse rotate predictions
    inverse_rot = rotation_matrix.T
    best_mode_displacements_tensor = torch.tensor(best_mode_displacements, dtype=torch.float32)
    best_mode_displacements_rot = torch.matmul(best_mode_displacements_tensor, inverse_rot.T).numpy()
    #print(f"[DEBUG] Best mode displacements (after inverse rotation):\n{best_mode_displacements_rot}")

     # ---  Reconstruct absolute (x, y) positions from displacements ---
    predicted_positions = []
    prev_x, prev_y = centers.numpy().flatten()
    for dx, dy in best_mode_displacements_rot:
        #future_x = round(prev_x + dx, 3)
        #future_y = round(prev_y + dy, 3)
        future_x = prev_x + dx
        future_y = prev_y + dy
        predicted_positions.append([future_x, future_y])
        prev_x, prev_y = future_x, future_y

    vehicle_positions_log[f"predicted_{vehicle.id}"] = predicted_positions

    print(f"[INFO] Vehicle {vehicle.id} ({vehicle.type_id}) at {timestamp}s -> Current Position: (X: {current_x:.3f}, Y: {current_y:.3f})")
    print(f"[INFO] Predicted positions for next timesteps: {predicted_positions[:5]}...")

    return predicted_positions


def store_vehicle_positions(timestamp, vehicles):
    """
    Stores the position of each vehicle at every second, starting from t = 3s.

    The function records each vehicle's (x, y) position in the `vehicle_positions_log`,
    allowing for real vs. predicted trajectory comparisons.

    Args:
        timestamp (int): The current simulation time in seconds.
        vehicles (list): A list of CARLA vehicle actors being tracked.

    Modifies:
        vehicle_positions_log (dict): Updates the log with vehicle positions at each timestamp.

    """
    if timestamp >= 3:  # Only store positions from t = 3s onward
        vehicle_positions_log[timestamp] = {}

        for vehicle in vehicles:
            transform = vehicle.get_transform()
            vehicle_positions_log[timestamp][vehicle.id] = [transform.location.x, transform.location.y]

        # print(f"Stored positions at timestamp {timestamp}s: {vehicle_positions_log[timestamp]}")  # Debug print


def cleanup_vehicles(vehicles_list, ego_vehicle):
    """Removes all spawned vehicles from the simulation."""
    print("\nRemoving all spawned vehicles...")
    for vehicle in vehicles_list:
        vehicle.destroy()

    if ego_vehicle:
        ego_vehicle.destroy()
        print("Ego Vehicle removed.")

    print("All vehicles removed.")


def main():
    """
    Main function to run CARLA simulation and integrate CRAT-Pred for trajectory prediction.

    This function:
        - Connects to CARLA.
        - Ensures asynchronous mode is enabled.
        - Loads the CRAT-Pred model.
        - Spawns the Ego Vehicle and NPC vehicles.
        - Stores vehicle positions and tracks their movements.
        - Runs CRAT-Pred to predict future trajectories.
        - Updates the spectator camera to follow the Ego Vehicle.
        - Cleans up vehicles upon simulation termination.
    """
    print("Connecting to CARLA...")

    try:
        world = connect_to_carla()
        print("Connected to CARLA.")
    except Exception as e:
        print(f"Error connecting to CARLA: {e}")
        return

    ensure_async_mode(world)

    # Retrieve available spawn points and vehicle blueprints
    spawn_points = get_spawn_points(world)
    blueprint_library = world.get_blueprint_library()

    if not spawn_points:
        print("No available spawn points. Exiting.")
        return

    print("Loading CRAT-Pred model...")
    model = load_cratpred_model()

    # Spawn Ego Vehicle
    ego_vehicle = spawn_ego_vehicle(world, spawn_points, blueprint_library)
    if ego_vehicle:
        transform = ego_vehicle.get_transform()
        vehicle_origins[ego_vehicle.id] = [transform.location.x, transform.location.y]

    # Spawn NPC vehicles
    num_npcs = 1
    vehicles = [ego_vehicle]
    vehicle_blueprints = [bp.id for bp in blueprint_library.filter("vehicle.*") if bp.has_attribute('number_of_wheels')]

    for i in range(min(num_npcs, len(spawn_points))):
        vehicle_type = random.choice(vehicle_blueprints)
        vehicle = spawn_vehicle(world, spawn_points, blueprint_library, vehicle_type)
        if vehicle:
            vehicles.append(vehicle)
            transform = vehicle.get_transform()
            vehicle_origins[vehicle.id] = [transform.location.x, transform.location.y]

    if not vehicles:
        print("No vehicles spawned. Exiting.")
        return

    print("\nAll vehicles spawned. Waiting 3 seconds for movement before predictions start...\n")
    time.sleep(3)

    for vehicle in vehicles:
        if vehicle.id not in vehicle_histories:
            vehicle_histories[vehicle.id] = []

    try:
        timestamp = 3
        while True:
            store_vehicle_positions(timestamp, vehicles)
            print(f"\nRunning vehicle tracking at {timestamp}s...")

            if timestamp == 3:
                for vehicle in vehicles:
                    transform = vehicle.get_transform()
                    vehicle_positions_log.setdefault(3, {})[vehicle.id] = [transform.location.x, transform.location.y]
                    print(f"[INFO] Vehicle {vehicle.id} ({vehicle.type_id}) at 3s -> Current Position: "
                          f"(X: {transform.location.x:.3f}, Y: {transform.location.y:.3f})")

            if timestamp > 3:
                for vehicle in vehicles:
                    if vehicle.id in vehicle_histories:
                        predict_future_cratpred(vehicle, model, timestamp)
                    else:
                        print(f"Warning: Vehicle {vehicle.id} not found in history. Skipping.")

            draw_predicted_trajectory(world, vehicles + [ego_vehicle], prediction_time=5, step_size=0.5)

            if ego_vehicle:
                update_spectator_view(world, ego_vehicle)

            timestamp += 1
            world.wait_for_tick()
            #time.sleep(1)  # Slows down simulation updates


    except KeyboardInterrupt:
        print("\nSimulation stopped.")

    finally:
        print("Cleaning up vehicles...")
        for vehicle in vehicles:
            if vehicle is not None:
                vehicle.destroy()  # Remove each spawned vehicle from the simulation
        print("All vehicles removed.")


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
















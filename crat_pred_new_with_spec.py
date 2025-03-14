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
vehicle_positions_log = {}  # Stores all recorded positions per second (timestamp → positions)
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
    Finds the best CRAT-Pred model checkpoint by selecting the one with the lowest validation loss.

    Args:
        checkpoint_dir (str): Path to the directory containing model checkpoint files.

    Returns:
        str: The path to the best checkpoint file.
    """
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]

    def extract_loss(filename):
        """Extracts the validation loss from the checkpoint filename."""
        try:
            return float(filename.split("loss_val=")[-1].replace(".ckpt", ""))
        except:
            return float("inf")  # Return a high value if extraction fails

    best_checkpoint = min(checkpoints, key=extract_loss)  # Select checkpoint with lowest validation loss
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
    Computes a more stable rotation matrix using a weighted moving average of past movement vectors.

    Args:
        vehicle_id (int): The unique ID of the vehicle.

    Returns:
        torch.Tensor: A (1, 2, 2) rotation matrix for aligning predictions.
                      Returns an identity matrix if insufficient data.
    """
    # Check if we have enough data points
    if vehicle_id not in vehicle_centers_matrix or vehicle_centers_matrix[vehicle_id].shape[1] < 5:
        return torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32).unsqueeze(0)

    # Convert stored positions to numpy
    centers_matrix_np = vehicle_centers_matrix[vehicle_id].squeeze(0).numpy()

    # **weighted moving average** over last 5 positions
    weights = np.array([1, 2, 3, 4, 5])  
    motion_vectors = np.diff(centers_matrix_np[-6:], axis=0)  # Last 5 motion vectors
    weighted_motion = np.average(motion_vectors, axis=0, weights=weights[:len(motion_vectors)])

    #  **Ensure non-zero motion vector**
    if np.linalg.norm(weighted_motion) > 0:  
        angle = np.arctan2(weighted_motion[1], weighted_motion[0])
    else:
        angle = 0  # Default to zero if no motion detected

    # Convert motion angle into a rotation matrix
    rotation_matrix = torch.tensor([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ], dtype=torch.float32).unsqueeze(0)

    return rotation_matrix



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
                carla.Location(waypoint_x, waypoint_y, vehicle.get_transform().location.z),
                "O",  # Waypoint symbol
                draw_shadow=False,
                color=color,
                life_time=1, 
                persistent_lines=False  # No connecting lines
            )

        #print(f"[INFO] Visualized {len(waypoints_to_draw)} waypoints for Vehicle {vehicle_id}")



def predict_future_cratpred(vehicle, model, timestamp):
    """
    Predicts future trajectories for a vehicle using the CRAT-Pred model.

    Args:
        vehicle (carla.Actor): The CARLA vehicle instance being tracked.
        model (torch.nn.Module): The trained CRAT-Pred model.
        timestamp (int): The current simulation timestamp in seconds.

    Returns:
        list: A list of predicted (x, y) positions for the next timesteps.
    """

    print(f"\n[INFO] Processing Vehicle {vehicle.type_id} (ID: {vehicle.id}) at {timestamp}s")

    # Retrieve vehicle's current position
    transform = vehicle.get_transform()
    current_x, current_y = round(transform.location.x, 3), round(transform.location.y, 3)

    # Store the first recorded position of the vehicle
    if vehicle.id not in vehicle_origins:
        vehicle_origins[vehicle.id] = [current_x, current_y]
        print(f"[DEBUG] Stored First Recorded Position for Vehicle {vehicle.id}: (X: {current_x}, Y: {current_y})")

    # Store position history
    vehicle_histories.setdefault(vehicle.id, []).append([current_x, current_y])

    # Maintain a max history of 21 positions (20 displacements)
    if len(vehicle_histories[vehicle.id]) > 21:
        vehicle_histories[vehicle.id].pop(0)

    # Retrieve past positions for trajectory tracking
    numeric_keys = sorted([t for t in vehicle_positions_log.keys() if isinstance(t, int)])
    centers_list = [vehicle_positions_log[t][vehicle.id] for t in numeric_keys if vehicle.id in vehicle_positions_log[t]]

    # Ensure at least one valid position exists
    if not centers_list:
        centers_list.append([current_x, current_y])

    # Convert to tensor format for model processing
    centers_matrix = torch.tensor(centers_list, dtype=torch.float32).unsqueeze(0)
    vehicle_centers_matrix[vehicle.id] = centers_matrix  

    # Compute displacements (change in position over time)
    centers_matrix_np = centers_matrix.squeeze(0).numpy()

    if timestamp == 3:  
        # No prediction at t=3s; we just record the position
        return  

    elif timestamp == 4:
        # Compute first displacement as (x4s - x3s, y4s - y3s)
        first_displacement = np.array([[centers_matrix_np[-1, 0] - centers_matrix_np[-2, 0], 
                                        centers_matrix_np[-1, 1] - centers_matrix_np[-2, 1]]])
        #print(f"[DEBUG] First displacement (x4s-x3s, y4s-y3s) for Vehicle {vehicle.id} at {timestamp}s: {first_displacement}")
        displacements = first_displacement  # Single row matrix for t=4s

    elif centers_matrix_np.shape[0] > 1:
        # Compute normal displacements (x_t - x_t-1, y_t - y_t-1) for t>=5s
        displacements = np.diff(centers_matrix_np, axis=0)

    else:
        # If only one recorded position, default to (0,0)
        displacements = np.array([[0.0, 0.0]])
        print(f"[DEBUG] Not enough history for displacement at {timestamp}s. Using (0,0) for Vehicle {vehicle.id}.")

    # Convert displacements to PyTorch tensor
    displacements_tensor = torch.tensor(displacements, dtype=torch.float32).unsqueeze(0)

    # Add a third feature with a value of 1 (flag for valid observations)
    ones_feature = torch.ones((displacements_tensor.shape[0], displacements_tensor.shape[1], 1), dtype=torch.float32)
    displacements_tensor = torch.cat((displacements_tensor, ones_feature), dim=-1)

    # Debugging: Print displacements every second
    #print(f"[DEBUG] Displacement Matrix for Vehicle {vehicle.id} at {timestamp}s:\n{displacements_tensor.numpy()}")

    # Determine origin (first recorded position at t=3s)
    if 3 in vehicle_positions_log and vehicle.id in vehicle_positions_log[3]:
        origin_x, origin_y = vehicle_positions_log[3][vehicle.id]
    #else:
    else:
         print(f"[ERROR] Missing t=3s position for Vehicle {vehicle.id}. Exiting prediction.")
         return None  # Exit function if t=3s data is missing

    origin = torch.tensor([origin_x, origin_y], dtype=torch.float32).view(1, 2)

    # Debugging: Print the origin position
    #print(f"[DEBUG] Origin Position for Vehicle {vehicle.id} at {timestamp}s: {origin.numpy()}")

    # Extract last observed position for trajectory input
    centers = centers_matrix[:, -1, :].view(1, 2)

    # Compute rotation matrix
    rotation_matrix = calculate_rotation(vehicle.id)

    # Prepare input batch for CRAT-Pred model
    batch = {
        "displ": (displacements_tensor,),
        "centers": (centers,),
        "rotation": rotation_matrix,
        "origin": origin
    }

    # Run prediction
    with torch.no_grad():
        predictions = model(batch)

    if predictions is None:
        print(f"[ERROR] CRAT-Pred returned None for Vehicle {vehicle.id}. Skipping prediction.")
        return None

    # Convert predictions to NumPy format
    predictions = predictions.squeeze(0).cpu().numpy()

    # Compute Final Displacement Error (FDE) across all predicted modes
    final_positions = predictions[0, :, -1]  
    real_last_x, real_last_y = centers.numpy().flatten()  
    fde_scores = np.linalg.norm(final_positions - [real_last_x, real_last_y], axis=1)

    # Debugging: Print extracted final positions
    #print(f"[DEBUG] Final Predicted Positions (Last Timesteps): \n{final_positions}")

    # Debugging: Print real last observed position
    #print(f"[DEBUG] Real Last Observed Position: (X: {real_last_x}, Y: {real_last_y})")

    # Debugging: Print computed FDE scores
    #print(f"[DEBUG] Computed FDE Scores for All Modes: {fde_scores}")

    # Select the best mode based on FDE
    best_mode_index = np.argmin(fde_scores)  

    # Extract the best predicted mode's displacements
    best_mode_displacements = predictions[0, best_mode_index]

    # Debugging: Print the best mode index
    #print(f"[DEBUG] Best Mode Index (Lowest FDE): {best_mode_index}")

    # Convert displacements into absolute future positions
    predicted_positions = []
    prev_x, prev_y = current_x, current_y

    for dx, dy in best_mode_displacements:
        future_x = round(prev_x + dx, 3)
        future_y = round(prev_y + dy, 3)
        predicted_positions.append([future_x, future_y])
        prev_x, prev_y = future_x, future_y

    # Store predicted positions
    vehicle_positions_log[f"predicted_{vehicle.id}"] = predicted_positions

    # Debugging: Predicted Positions
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

    Runs indefinitely until manually stopped (KeyboardInterrupt).

    Exits:
        If CARLA fails to connect or no vehicles are successfully spawned.
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

    # Spawn Ego Vehicle (Main controlled vehicle)
    ego_vehicle = spawn_ego_vehicle(world, spawn_points, blueprint_library)

    # Store ego vehicle's origin position for trajectory tracking
    if ego_vehicle:
        transform = ego_vehicle.get_transform()
        vehicle_origins[ego_vehicle.id] = [transform.location.x, transform.location.y]

    # Spawn NPC vehicles (Non-player characters)
    num_npcs = 1  # Increase to 10 for better visualization
    vehicles = [ego_vehicle]  # Start the list with the Ego Vehicle
    vehicle_blueprints = [bp.id for bp in blueprint_library.filter("vehicle.*") if bp.has_attribute('number_of_wheels')]

    for i in range(min(num_npcs, len(spawn_points))):
        vehicle_type = random.choice(vehicle_blueprints)  # Select a random vehicle type
        vehicle = spawn_vehicle(world, spawn_points, blueprint_library, vehicle_type)
        if vehicle:
            vehicles.append(vehicle)
            # Store NPC vehicle's origin position
            transform = vehicle.get_transform()
            vehicle_origins[vehicle.id] = [transform.location.x, transform.location.y]

    if not vehicles:
        print("No vehicles spawned. Exiting.")
        return

    print("\nAll vehicles spawned. Waiting 3 seconds for movement before predictions start...\n")
    time.sleep(3)  # Allow vehicles to move before starting tracking

    # Ensure all vehicles have a history tracking entry
    for vehicle in vehicles:
        if vehicle.id not in vehicle_histories:
            vehicle_histories[vehicle.id] = []

    try:
        timestamp = 3
        while True:
            store_vehicle_positions(timestamp, vehicles)  # Log real-time positions
            print(f"\nRunning vehicle tracking at {timestamp}s...")

            if timestamp == 3:
                for vehicle in vehicles:
                    transform = vehicle.get_transform()
                    vehicle_positions_log.setdefault(3, {})[vehicle.id] = [transform.location.x, transform.location.y]
                    print(f"[INFO] Vehicle {vehicle.id} ({vehicle.type_id}) at 3s -> Current Position: (X: {transform.location.x:.3f}, Y: {transform.location.y:.3f})")

            # Skip calling predict_future_cratpred at t=3s
            if timestamp > 3:
                for vehicle in vehicles:
                    if vehicle.id in vehicle_histories:
                        predict_future_cratpred(vehicle, model, timestamp)  # Run trajectory prediction
                    else:
                        print(f"Warning: Vehicle {vehicle.id} not found in history. Skipping.")

            draw_predicted_trajectory(world, vehicles + [ego_vehicle], prediction_time=5, step_size=0.5)

            # Update spectator camera to follow Ego Vehicle
            if ego_vehicle:
                update_spectator_view(world, ego_vehicle)

            timestamp += 1
            time.sleep(1.0)  # Wait 1 second before next tracking cycle

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









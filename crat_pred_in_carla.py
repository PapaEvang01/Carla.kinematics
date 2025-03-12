import time
import random
import numpy as np
import torch
import carla
import sys
import os

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


def calculate_rotation(vehicle_id):
    """
    Calculates the rotation matrix for a vehicle based on its last two recorded positions.

    Args:
        vehicle_id (int): The unique ID of the vehicle.

    Returns:
        torch.Tensor: A (1, 2, 2) rotation matrix representing the vehicle's orientation.
                      Returns an identity matrix if there is insufficient movement data.
    """
    if vehicle_id not in vehicle_centers_matrix or vehicle_centers_matrix[vehicle_id].shape[1] < 2:
        return torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32).unsqueeze(0)  # Default identity matrix
    
    centers_matrix_np = vehicle_centers_matrix[vehicle_id].squeeze(0).numpy()
    last_vector = centers_matrix_np[-1] - centers_matrix_np[-2]  # Compute the direction of the last movement
    angle = np.arctan2(last_vector[1], last_vector[0])  # Compute the angle of motion

    rotation_matrix = torch.tensor([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ], dtype=torch.float32).unsqueeze(0)

    #print(f"Computed Rotation Matrix for Vehicle {vehicle_id}: \n{rotation_matrix}")  # Debug print

    return rotation_matrix


def predict_future_cratpred(vehicle, model, timestamp):
    """
    Predicts future trajectories for a vehicle using the CRAT-Pred model.

    The function retrieves the vehicle's current position, computes its movement history,
    and feeds this data into the CRAT-Pred model. The model predicts future displacements,
    and the function selects the best mode dynamically based on Final Displacement Error (FDE).

    Args:
        vehicle (carla.Actor): The CARLA vehicle instance being tracked.
        model (torch.nn.Module): The trained CRAT-Pred model.
        timestamp (int): The current simulation timestamp in seconds.

    Returns:
        list: A list of predicted (x, y) positions for the next timesteps.
    """

    print(f"\nProcessing vehicle: {vehicle.type_id} (ID: {vehicle.id})")  # Debug print

    # Retrieve the vehicle's current position
    transform = vehicle.get_transform()
    current_x, current_y = round(transform.location.x, 3), round(transform.location.y, 3)
    vehicle_histories.setdefault(vehicle.id, []).append([current_x, current_y])

    # Maintain a max history of 21 positions (20 displacements for tracking movement trends)
    if len(vehicle_histories[vehicle.id]) > 21:
        vehicle_histories[vehicle.id].pop(0)

    # Extract only numeric timestamps from the vehicle positions log
    numeric_keys = [t for t in vehicle_positions_log.keys() if isinstance(t, int)]
    centers_list = []

    # Retrieve past positions for trajectory tracking
    for t in sorted(numeric_keys):
        if vehicle.id in vehicle_positions_log[t]:
            centers_list.append(vehicle_positions_log[t][vehicle.id])

    # Ensure at least one valid position exists
    if not centers_list:
        centers_list.append([current_x, current_y])

    # Convert to tensor format for model processing
    centers_matrix = torch.tensor(centers_list, dtype=torch.float32).unsqueeze(0)
    vehicle_centers_matrix[vehicle.id] = centers_matrix  # Store centers for future use

    # Compute displacements (change in position over time)
    centers_matrix_np = centers_matrix.squeeze(0).numpy()
    if centers_matrix_np.shape[0] >= 1:
        displacements = np.diff(centers_matrix_np, axis=0)
        displacements = np.vstack((centers_matrix_np[0], displacements))
    else:
        displacements = np.array([[current_x, current_y]])  # Default displacement (no movement history)

    # Convert displacements to PyTorch tensors
    displacements_tensor = torch.tensor(displacements, dtype=torch.float32).unsqueeze(0)

    # Add a zero third feature to match the model's expected input shape (1, T, 3)
    zero_feature = torch.zeros((displacements_tensor.shape[0], displacements_tensor.shape[1], 1), dtype=torch.float32)
    displacements_tensor = torch.cat((displacements_tensor, zero_feature), dim=-1)

    # Set origin as the first recorded position
    origin = torch.tensor(vehicle_origins[vehicle.id], dtype=torch.float32).view(1, 2)

    # Extract last position for current prediction input
    centers = centers_matrix[:, -1, :].view(1, 2)

    # Compute rotation matrix
    rotation_matrix = calculate_rotation(vehicle.id)

    # Debug prints
    # print(f"Origin for Vehicle {vehicle.id}: (X: {origin[0, 0]:.3f}, Y: {origin[0, 1]:.3f})")
    # print(f"Centers for Vehicle {vehicle.id}: (X: {centers[0, 0]:.3f}, Y: {centers[0, 1]:.3f})")

    # Prepare input batch for the model
    batch = {
        "displ": (displacements_tensor,),
        "centers": (centers,),
        "rotation": rotation_matrix,
        "origin": origin
    }

    # print("Batch prepared. Running model prediction...")  # Debug print

    # Run prediction
    with torch.no_grad():
        predictions = model(batch)

    if predictions is None:
        print(f"CRAT-Pred returned None for Vehicle {vehicle.id}. Skipping prediction.")
        return None

    # Convert model output to NumPy format
    predictions = predictions.squeeze(0).cpu().numpy()
    # print(f"\nModel output shape: {predictions.shape}")  # Debug print

    # Compute Final Displacement Error (FDE) for all modes
    final_positions = predictions[0, :, -1]  # Last (x, y) position of each mode
    real_last_x, real_last_y = centers.numpy().flatten()  # Last observed position
    fde_scores = np.linalg.norm(final_positions - [real_last_x, real_last_y], axis=1)

    # print(f"FDE scores for Vehicle {vehicle.id}: {np.round(fde_scores, 3)}")  # Debug print

    # Choose the best mode dynamically based on FDE
    if np.std(fde_scores) < 0.5:  # If all FDE scores are very close, randomize selection slightly
        best_mode_index = np.random.choice(np.argsort(fde_scores)[:3])  # Choose from top 3
        # print(f"FDE scores are close. Randomly selected best mode from top 3: Mode {best_mode_index}")  # Debug print
    else:
        best_mode_index = np.argmin(fde_scores)  # Choose the mode with the lowest FDE

    # print(f"Best mode selected for Vehicle {vehicle.id}: Mode {best_mode_index}")  # Debug print

    # Extract the best mode's future displacements
    best_mode_displacements = predictions[0, best_mode_index]

    # print(f"Best mode displacements: {np.round(best_mode_displacements[:5], 3)}...")  # Debug print

    # Convert displacements into absolute future positions
    predicted_positions = []
    prev_x, prev_y = current_x, current_y

    for dx, dy in best_mode_displacements:
        future_x = round(prev_x + dx, 3)
        future_y = round(prev_y + dy, 3)
        predicted_positions.append([future_x, future_y])
        prev_x, prev_y = future_x, future_y

    # Store the predicted future positions for later use
    vehicle_positions_log[f"predicted_{vehicle.id}"] = predicted_positions

    # Print the current and predicted positions
    print(f"Vehicle {vehicle.id} ({vehicle.type_id}) at {timestamp}s -> Current Position: (X: {current_x:.3f}, Y: {current_y:.3f})")
    print(f"Predicted positions for next timesteps: {predicted_positions[:5]}...")  # Print first 5 for reference

    # Print predicted position for the next second separately
    if len(predicted_positions) > 1:
        next_x, next_y = predicted_positions[1]  # The second timestep is the next second
        print(f"Vehicle {vehicle.id} ({vehicle.type_id}) predicted position at {timestamp+1}s -> (X: {next_x:.3f}, Y: {next_y:.3f})")

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
        - Spawns the ego vehicle and NPC vehicles.
        - Stores vehicle positions and tracks their movements.
        - Runs CRAT-Pred to predict future trajectories.
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
    num_npcs = 1  # Number of NPCs to spawn
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

            for vehicle in vehicles:
                if vehicle.id in vehicle_histories:
                    predict_future_cratpred(vehicle, model, timestamp)  # Run trajectory prediction
                else:
                    print(f"Warning: Vehicle {vehicle.id} not found in history. Skipping.")

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

if __name__ == "__main__":
    main()











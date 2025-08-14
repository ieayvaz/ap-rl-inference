import socket
import json
import threading
import time
import numpy as np
from stable_baselines3 import PPO
import sys

# State bounds for normalization (same as training)
STATE_BOUNDS = {
    'distance': (0, 1000),
    'distance_x': (-1000, 1000),
    'distance_y': (-1000, 1000),
    'distance_z': (-250, 250),
    'own_roll_sin': (-1, 1),
    'own_roll_cos': (-1, 1),
    'own_pitch_sin': (-1, 1),
    'own_pitch_cos': (-1, 1),
    'own_heading_sin': (-1, 1),
    'own_heading_cos': (-1, 1),
    'enemy_roll_sin': (-1, 1),
    'enemy_roll_cos': (-1, 1),
    'enemy_pitch_sin': (-1, 1),
    'enemy_pitch_cos': (-1, 1),
    'enemy_heading_sin': (-1, 1),
    'enemy_heading_cos': (-1, 1),
    'relative_velocity_x': (-50, 50),
    'relative_velocity_y': (-50, 50),
    'relative_velocity_z': (-25, 25),
    'q_radps': (-1, 1),
    'p_radps': (-1, 1),
    'r_radps': (-1, 1),
    'current_lock_duration': (0, 10),
    'current_lock_status': (0, 1),
    'target_roll': (-60, 60),
    'target_pitch': (-25, 25),
    'target_throttle': (0, 1),
    'los_error_azimuth_cos': (-1, 1),
    'los_error_azimuth_sin': (-1, 1),
    'los_error_elev_cos': (-1, 1),
    'los_error_elev_sin': (-1, 1),
    'prev_roll_pid_dterm': (-60, 60),
    'prev_roll_pid_iterm': (-60, 60),
    'prev_roll_pid_pterm': (-60, 60),
    'prev_pitch_pid_pterm': (-25, 25),
    'prev_pitch_pid_iterm': (-25, 25),
    'prev_pitch_pid_dterm': (-25, 25)
}

def normalize_state(state):
    """
    Normalize state vector to [-1, 1] range using training bounds
    
    Args:
        state: list or numpy array of raw state values (39 elements)
    
    Returns:
        numpy array of normalized state values
    """
    state = np.array(state, dtype=np.float32)
    
    # State order must match the order in get_state function
    state_names = [
        'distance', 'distance_x', 'distance_y', 'distance_z',
        'own_roll_sin', 'own_roll_cos', 'own_pitch_sin', 'own_pitch_cos',
        'own_heading_sin', 'own_heading_cos',
        'enemy_roll_sin', 'enemy_roll_cos', 'enemy_pitch_sin', 'enemy_pitch_cos',
        'enemy_heading_sin', 'enemy_heading_cos',
        'relative_velocity_x', 'relative_velocity_y', 'relative_velocity_z',
        'q_radps', 'p_radps', 'r_radps',
        'current_lock_duration', 'current_lock_status',
        'target_roll', 'target_pitch', 'target_throttle',
        'los_error_azimuth_cos', 'los_error_azimuth_sin',
        'los_error_elev_cos', 'los_error_elev_sin',
        'prev_roll_pid_dterm', 'prev_roll_pid_iterm', 'prev_roll_pid_pterm',
        'prev_pitch_pid_pterm', 'prev_pitch_pid_iterm', 'prev_pitch_pid_dterm'
    ]
    
    normalized_state = np.zeros_like(state)
    
    for i, (value, name) in enumerate(zip(state, state_names)):
        min_val, max_val = STATE_BOUNDS[name]
        
        # Clip value to bounds
        clipped_value = np.clip(value, min_val, max_val)
        
        # Normalize to [-1, 1] range
        if max_val > min_val:
            normalized_value = 2 * (clipped_value - min_val) / (max_val - min_val) - 1
        else:
            normalized_value = 0
        
        normalized_state[i] = normalized_value
    
    return normalized_state.astype(np.float32)

class GroundStationServer:
    """Server that runs the ML model and sends commands to UAV"""
    
    def __init__(self, model_path, host='0.0.0.0', port=5555):
        """
        Initialize the ground station server
        
        Args:
            model_path: Path to the trained PPO model
            host: Host IP to bind to (0.0.0.0 for all interfaces)
            port: Port to listen on
        """
        self.model_path = model_path
        self.host = host
        self.port = port
        self.model = None
        self.server_socket = None
        self.running = False
        self.clients = {}
        self.client_counter = 0
        
        # Statistics
        self.total_inferences = 0
        self.start_time = None
        
    def load_model(self):
        """Load the PPO model"""
        print(f"Loading model from {self.model_path}...")
        self.model = PPO.load(self.model_path)
        print("Model loaded successfully!")
    
    def start_server(self):
        """Start the TCP server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True
        self.start_time = time.time()
        
        print(f"Ground Station Server listening on {self.host}:{self.port}")
        
        # Start accepting clients
        accept_thread = threading.Thread(target=self.accept_clients)
        accept_thread.daemon = True
        accept_thread.start()
        
        # Start statistics thread
        stats_thread = threading.Thread(target=self.print_statistics)
        stats_thread.daemon = True
        stats_thread.start()
    
    def accept_clients(self):
        """Accept incoming client connections"""
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                client_id = self.client_counter
                self.client_counter += 1
                
                self.clients[client_id] = {
                    'socket': client_socket,
                    'address': address,
                    'connected_at': time.time(),
                    'inferences': 0
                }
                
                print(f"New UAV client connected from {address} (ID: {client_id})")
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_id,)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"Error accepting client: {e}")
    
    def handle_client(self, client_id):
        """Handle communication with a single UAV client"""
        client = self.clients[client_id]
        client_socket = client['socket']
        
        try:
            while self.running:
                # Receive message length
                length_bytes = client_socket.recv(4)
                if not length_bytes:
                    break
                
                message_length = int.from_bytes(length_bytes, 'big')
                
                # Receive message
                message_bytes = client_socket.recv(message_length)
                if not message_bytes:
                    break
                
                message = json.loads(message_bytes.decode('utf-8'))
                
                # Process message
                if message['type'] == 'state':
                    # Get state and run inference
                    state = message['state']
                    action = self.get_action(state)
                    
                    # Send action back
                    response = {
                        'type': 'action',
                        'action': action.tolist() if isinstance(action, np.ndarray) else action,
                        'timestamp': time.time()
                    }
                    
                    json_response = json.dumps(response)
                    response_bytes = json_response.encode('utf-8')
                    client_socket.sendall(len(response_bytes).to_bytes(4, 'big'))
                    client_socket.sendall(response_bytes)
                    
                    # Update statistics
                    client['inferences'] += 1
                    self.total_inferences += 1
                    
                elif message['type'] == 'disconnect':
                    print(f"Client {client_id} requested disconnect")
                    break
                    
        except Exception as e:
            print(f"Error handling client {client_id}: {e}")
        finally:
            # Clean up
            client_socket.close()
            del self.clients[client_id]
            print(f"Client {client_id} disconnected")
    
    def get_action(self, state):
        """
        Run inference on the model
        
        Args:
            state: List of state values (39 elements)
        
        Returns:
            Action array [roll_idx, pitch_idx, throttle_idx]
        """
        # Normalize state
        normalized_state = normalize_state(state)
        
        # Run inference
        action, _ = self.model.predict(normalized_state, deterministic=False)
        
        return action
    
    def print_statistics(self):
        """Print server statistics periodically"""
        while self.running:
            time.sleep(10)  # Print every 10 seconds
            
            if self.start_time:
                uptime = time.time() - self.start_time
                inference_rate = self.total_inferences / uptime if uptime > 0 else 0
                
                print("\n=== Ground Station Statistics ===")
                print(f"Uptime: {uptime:.1f} seconds")
                print(f"Active clients: {len(self.clients)}")
                print(f"Total inferences: {self.total_inferences}")
                print(f"Inference rate: {inference_rate:.2f} Hz")
                
                for client_id, client in self.clients.items():
                    client_uptime = time.time() - client['connected_at']
                    client_rate = client['inferences'] / client_uptime if client_uptime > 0 else 0
                    print(f"  Client {client_id} ({client['address']}): "
                          f"{client['inferences']} inferences, {client_rate:.2f} Hz")
                print("================================\n")
    
    def stop_server(self):
        """Stop the server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        print("Server stopped")

def main():
    if len(sys.argv) != 2:
        print("Usage: python ground_station_server.py <model_path>")
        print("Example: python ground_station_server.py model.zip")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Create and start server
    server = GroundStationServer(model_path, host='0.0.0.0', port=5555)
    
    try:
        # Load model
        server.load_model()
        
        # Start server
        server.start_server()
        
        print("\nGround Station Server is running...")
        print("Press Ctrl+C to stop\n")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop_server()
    except Exception as e:
        print(f"Error: {e}")
        server.stop_server()

if __name__ == "__main__":
    main()
"""
Enemy Aircraft Data Server
Connects to local SITL and broadcasts enemy aircraft data to network clients
"""

import socket
import json
import time
import threading
from dronekit import connect, VehicleMode
import sys

class EnemyDataServer:
    """Server that broadcasts enemy aircraft data to network clients"""
    
    def __init__(self, sitl_address, host='0.0.0.0', port=6666):
        """
        Initialize the enemy data server
        
        Args:
            sitl_address: MAVLink address of enemy SITL (e.g., '127.0.0.1:14551')
            host: Host IP to bind to (0.0.0.0 for all interfaces)
            port: Port to listen on for data requests
        """
        self.sitl_address = sitl_address
        self.host = host
        self.port = port
        self.vehicle = None
        self.server_socket = None
        self.running = False
        self.clients = []
        self.data_lock = threading.Lock()
        self.latest_data = None
        
        # Statistics
        self.total_requests = 0
        self.start_time = None
        
    def connect_to_vehicle(self):
        """Connect to the enemy vehicle via MAVLink"""
        print(f"Connecting to enemy vehicle at {self.sitl_address}...")
        self.vehicle = connect(self.sitl_address, wait_ready=True)
        print(f"Enemy vehicle connected! Mode: {self.vehicle.mode.name}")
        print(f"  Location: {self.vehicle.location.global_frame}")
        print(f"  Heading: {self.vehicle.heading}°")
        
    def update_vehicle_data(self):
        """Continuously update vehicle data from MAVLink"""
        while self.running:
            try:
                if self.vehicle and self.vehicle.location.global_frame:
                    data = {
                        'lat': self.vehicle.location.global_frame.lat,
                        'lon': self.vehicle.location.global_frame.lon,
                        'alt': self.vehicle.location.global_frame.alt,
                        'roll': self.vehicle.attitude.roll,
                        'pitch': self.vehicle.attitude.pitch,
                        'yaw': self.vehicle.attitude.yaw,
                        'heading': self.vehicle.heading,
                        'velocity': list(self.vehicle.velocity),  # [vx, vy, vz]
                        'groundspeed': self.vehicle.groundspeed,
                        'airspeed': self.vehicle.airspeed,
                        'mode': self.vehicle.mode.name,
                        'armed': self.vehicle.armed,
                        'timestamp': time.time()
                    }
                    
                    with self.data_lock:
                        self.latest_data = data
                        
                time.sleep(0.05)  # Update at 20Hz
                
            except Exception as e:
                print(f"Error updating vehicle data: {e}")
                time.sleep(1)
    
    def start_server(self):
        """Start the TCP server for client requests"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True
        self.start_time = time.time()
        
        print(f"\nEnemy Data Server listening on {self.host}:{self.port}")
        print("Waiting for client connections...\n")
        
        # Start vehicle data update thread
        update_thread = threading.Thread(target=self.update_vehicle_data)
        update_thread.daemon = True
        update_thread.start()
        
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
                print(f"New client connected from {address}")
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"Error accepting client: {e}")
    
    def handle_client(self, client_socket, address):
        """Handle communication with a single client"""
        try:
            while self.running:
                # Receive request
                try:
                    request_bytes = client_socket.recv(1024)
                    if not request_bytes:
                        break
                    
                    request = json.loads(request_bytes.decode('utf-8'))
                    
                    if request.get('type') == 'get_data':
                        # Send latest enemy data
                        with self.data_lock:
                            if self.latest_data:
                                response = {
                                    'type': 'enemy_data',
                                    'data': self.latest_data,
                                    'server_time': time.time()
                                }
                            else:
                                response = {
                                    'type': 'error',
                                    'message': 'No data available yet',
                                    'server_time': time.time()
                                }
                        
                        json_response = json.dumps(response)
                        client_socket.sendall(json_response.encode('utf-8'))
                        self.total_requests += 1
                        
                    elif request.get('type') == 'disconnect':
                        print(f"Client {address} requested disconnect")
                        break
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Error handling request from {address}: {e}")
                    break
                    
        except Exception as e:
            print(f"Error with client {address}: {e}")
        finally:
            client_socket.close()
            print(f"Client {address} disconnected")
    
    def print_statistics(self):
        """Print server statistics periodically"""
        while self.running:
            time.sleep(30)  # Print every 30 seconds
            
            if self.start_time and self.latest_data:
                uptime = time.time() - self.start_time
                request_rate = self.total_requests / uptime if uptime > 0 else 0
                
                print("\n=== Enemy Server Statistics ===")
                print(f"Uptime: {uptime:.1f} seconds")
                print(f"Total requests served: {self.total_requests}")
                print(f"Request rate: {request_rate:.2f} Hz")
                
                with self.data_lock:
                    if self.latest_data:
                        print(f"\nEnemy Aircraft Status:")
                        print(f"  Position: ({self.latest_data['lat']:.6f}, {self.latest_data['lon']:.6f}, {self.latest_data['alt']:.1f}m)")
                        print(f"  Heading: {self.latest_data['heading']:.1f}°")
                        print(f"  Speed: {self.latest_data['groundspeed']:.1f} m/s")
                        print(f"  Mode: {self.latest_data['mode']}")
                        print(f"  Armed: {self.latest_data['armed']}")
                print("================================\n")
    
    def stop_server(self):
        """Stop the server and disconnect from vehicle"""
        self.running = False
        
        if self.server_socket:
            self.server_socket.close()
            
        if self.vehicle:
            self.vehicle.close()
            print("Disconnected from enemy vehicle")
            
        print("Enemy server stopped")

def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python enemy_server.py <sitl_address> [host] [port]")
        print("Example: python enemy_server.py 127.0.0.1:14551")
        print("Example: python enemy_server.py 127.0.0.1:14551 0.0.0.0 6666")
        sys.exit(1)
    
    sitl_address = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) > 2 else '0.0.0.0'
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 6666
    
    # Create and start server
    server = EnemyDataServer(sitl_address, host, port)
    
    try:
        # Connect to vehicle
        server.connect_to_vehicle()
        
        # Start server
        server.start_server()
        
        print("\nEnemy Data Server is running...")
        print(f"Other computers can connect to: {socket.gethostbyname(socket.gethostname())}:{port}")
        print("Press Ctrl+C to stop\n")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down enemy server...")
        server.stop_server()
    except Exception as e:
        print(f"Error: {e}")
        server.stop_server()

if __name__ == "__main__":
    main()
import math
import sys
import time
import socket
import json
import numpy as np
from dronekit import connect, VehicleMode
from geopy.distance import distance as geo_distance

# All params (same as original)
ROLL_MIN = -50
ROLL_MAX = 50
PITCH_MIN = -20
PITCH_MAX = 20
THROTTLE_MAX = 0.6
THROTTLE_MIN = 0.4
ROLL_ACTION_DELTAS = [30.0, 20.0, 10.0, 5, 0, -5, -10.0, -20.0, -30.0]
PITCH_ACTION_DELTAS = [3, 1.5, 0, -1.5, -3]
THROTTLE_ACTION_DELTAS = [0.05, 0, -0.05]
ROLL_CHANNEL_MAX = 1900
ROLL_CHANNEL_MIN = 1100
PITCH_CHANNEL_MAX = 1900
PITCH_CHANNEL_MIN = 1100
PITCH_CHANNEL_TRIM = 1500
THROTTLE_CHANNEL_TRIM = 1500
THROTTLE_CHANNEL_MAX = 1900
THROTTLE_CHANNEL_MIN = 1100


def connect_mavlink(address):
    return connect(address, wait_ready=True)

class PID:
    """PID Controller from IvPID"""
    def __init__(self, P=0.2, I=0.0, D=0.0, current_time=None):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        
        self.sample_time = 0.00
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time
        
        self.clear()
    
    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0
        
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        
        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0
        
        self.output = 0.0
    
    def update(self, feedback_value, current_time=None):
        """Calculates PID value for given reference feedback"""
        error = self.SetPoint - feedback_value
        
        self.current_time = current_time if current_time is not None else time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error
        
        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time
            
            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard
            
            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time
            
            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error
            
            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)
    
    def setWindup(self, windup):
        """Set integral windup guard"""
        self.windup_guard = windup
    
    def setSampleTime(self, sample_time):
        """Set PID sample time"""
        self.sample_time = sample_time

# Global variables for enemy tracking
enemy_vehicle = None
enemy_server_url = None
visualization_origin = None
lock_start_time = None
current_lock_duration = 0.0

def lla_to_ecef(lat, lon, alt):
    """Convert LLA to ECEF"""
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f * f
    
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
    
    x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (N * (1 - e2) + alt) * math.sin(lat_rad)
    
    return np.array([x, y, z])

def ecef_to_enu(ecef_point, ref_lat, ref_lon, ref_alt):
    """Convert ECEF to ENU"""
    ref_ecef = lla_to_ecef(ref_lat, ref_lon, ref_alt)
    delta_ecef = ecef_point - ref_ecef
    
    lat_rad = math.radians(ref_lat)
    lon_rad = math.radians(ref_lon)
    
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    
    R = np.array([
        [-sin_lon,           cos_lon,          0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon,  cos_lat * sin_lon,  sin_lat]
    ])
    
    enu = R @ delta_ecef
    return enu

def lla_to_enu(target_lat, target_lon, target_alt, ref_lat, ref_lon, ref_alt):
    """Direct conversion from LLA to ENU"""
    target_ecef = lla_to_ecef(target_lat, target_lon, target_alt)
    return ecef_to_enu(target_ecef, ref_lat, ref_lon, ref_alt)

def setup_enemy_tracking(method='mavlink', address=None):
    """Setup enemy tracking method"""
    global enemy_vehicle, enemy_server_url
    
    if method == 'mavlink' and address:
        enemy_vehicle = connect(address, wait_ready=True)
        print(f"Connected to enemy vehicle at {address}")
    elif method == 'server' and address:
        enemy_server_url = address
        print(f"Will fetch enemy data from server: {address}")

def get_enemy_data_from_mavlink():
    """Get enemy aircraft data from MAVLink connection"""
    if enemy_vehicle is None:
        return None
    
    return {
        'lat': enemy_vehicle.location.global_frame.lat,
        'lon': enemy_vehicle.location.global_frame.lon,
        'alt': enemy_vehicle.location.global_frame.alt,
        'roll': enemy_vehicle.attitude.roll,
        'pitch': enemy_vehicle.attitude.pitch,
        'heading': enemy_vehicle.heading,
        'velocity': enemy_vehicle.velocity
    }

def calculate_los_angles_and_errors(enu_vector, own_heading, own_pitch):
    """Calculate Line of Sight angles and errors"""
    east = enu_vector[0]
    north = enu_vector[1]
    up = enu_vector[2]
    
    angle_rad = math.atan2(north, east)
    angle_deg = math.degrees(angle_rad)
    los_azimuth = (90 - angle_deg + 360) % 360
    
    horizontal_dist = math.sqrt(east**2 + north**2)
    elevation_rad = math.atan2(up, horizontal_dist)
    los_elevation = math.degrees(elevation_rad)
    
    azimuth_error = los_azimuth - own_heading
    if azimuth_error > 180:
        azimuth_error -= 360
    elif azimuth_error < -180:
        azimuth_error += 360
    
    own_pitch_deg = math.degrees(own_pitch)
    elevation_error = los_elevation - own_pitch_deg
    
    return los_azimuth, los_elevation, azimuth_error, elevation_error

def check_lock_status(distance_3d, azimuth_error, elevation_error):
    """Check if locked onto target"""
    if distance_3d > 50 or distance_3d < 10:
        return False
    if abs(azimuth_error) > 25:
        return False
    if abs(elevation_error) > 15:
        return False
    return True

def get_state_for_ground_station(vehicle, target_attitude, pid_roll, pid_pitch):
    """
    Extract state observation to send to ground station for inference
    Returns state as list for JSON serialization
    """
    global lock_start_time, current_lock_duration, visualization_origin
    
    # Get own aircraft data
    own_lat = vehicle.location.global_frame.lat
    own_lon = vehicle.location.global_frame.lon
    own_alt = vehicle.location.global_frame.alt
    own_roll = vehicle.attitude.roll
    own_pitch = vehicle.attitude.pitch
    own_heading = vehicle.heading
    own_velocity = vehicle.velocity
    
    # Set origin on first call
    if visualization_origin is None:
        visualization_origin = (own_lat, own_lon, own_alt)
        print(f"Origin set to: Lat={own_lat:.6f}, Lon={own_lon:.6f}, Alt={own_alt:.1f}m")
    
    # Get enemy aircraft data
    if enemy_vehicle is not None:
        enemy_data = get_enemy_data_from_mavlink()
    else:
        enemy_data = {
            'lat': own_lat,
            'lon': own_lon,
            'alt': own_alt,
            'roll': 0,
            'pitch': 0,
            'heading': 0,
            'velocity': [0, 0, 0]
        }
    
    if enemy_data is None:
        return None
    
    # Calculate relative ENU
    enu_enemy_relative = lla_to_enu(
        enemy_data['lat'], enemy_data['lon'], enemy_data['alt'],
        own_lat, own_lon, own_alt
    )
    
    distance_east = enu_enemy_relative[0]
    distance_north = enu_enemy_relative[1]
    distance_up = enu_enemy_relative[2]
    
    distance_x = distance_east
    distance_y = distance_north
    distance_z = distance_up
    
    distance_3d = math.sqrt(distance_x**2 + distance_y**2 + distance_z**2)
    
    # Trigonometric features
    own_roll_sin = math.sin(own_roll)
    own_roll_cos = math.cos(own_roll)
    own_pitch_sin = math.sin(own_pitch)
    own_pitch_cos = math.cos(own_pitch)
    own_heading_rad = math.radians(own_heading)
    own_heading_sin = math.sin(own_heading_rad)
    own_heading_cos = math.cos(own_heading_rad)
    
    enemy_roll_sin = math.sin(enemy_data['roll'])
    enemy_roll_cos = math.cos(enemy_data['roll'])
    enemy_pitch_sin = math.sin(enemy_data['pitch'])
    enemy_pitch_cos = math.cos(enemy_data['pitch'])
    enemy_heading_rad = math.radians(enemy_data['heading'])
    enemy_heading_sin = math.sin(enemy_heading_rad)
    enemy_heading_cos = math.cos(enemy_heading_rad)
    
    # Relative velocities
    relative_velocity_x = enemy_data['velocity'][0] - own_velocity[0]
    relative_velocity_y = enemy_data['velocity'][1] - own_velocity[1]
    relative_velocity_z = enemy_data['velocity'][2] - own_velocity[2]
    
    # Angular velocities
    try:
        p_radps = vehicle.raw_imu.xgyro / 1000.0
        q_radps = vehicle.raw_imu.ygyro / 1000.0
        r_radps = vehicle.raw_imu.zgyro / 1000.0
    except:
        p_radps = 0.0
        q_radps = 0.0
        r_radps = 0.0
    
    # Calculate LOS angles and errors
    los_azimuth, los_elevation, azimuth_error_deg, elevation_error_deg = \
        calculate_los_angles_and_errors(enu_enemy_relative, own_heading, own_pitch)
    
    los_azimuth_error_rad = math.radians(azimuth_error_deg)
    los_elevation_error_rad = math.radians(elevation_error_deg)
    
    los_error_azimuth_cos = math.cos(los_azimuth_error_rad)
    los_error_azimuth_sin = math.sin(los_azimuth_error_rad)
    los_error_elev_cos = math.cos(los_elevation_error_rad)
    los_error_elev_sin = math.sin(los_elevation_error_rad)
    
    # Lock status
    is_locked = check_lock_status(distance_3d, azimuth_error_deg, elevation_error_deg)
    current_lock_status = 1.0 if is_locked else 0.0
    
    # Update lock duration
    if is_locked:
        if lock_start_time is None:
            lock_start_time = time.time()
        current_lock_duration = time.time() - lock_start_time
    else:
        lock_start_time = None
        current_lock_duration = 0.0
    
    # Target attitude values
    target_roll = target_attitude[0]
    target_pitch = target_attitude[1]
    target_throttle = target_attitude[2]
    
    # PID terms
    prev_roll_pid_pterm = pid_roll.PTerm
    prev_roll_pid_iterm = pid_roll.ITerm
    prev_roll_pid_dterm = pid_roll.DTerm
    prev_pitch_pid_pterm = pid_pitch.PTerm
    prev_pitch_pid_iterm = pid_pitch.ITerm
    prev_pitch_pid_dterm = pid_pitch.DTerm
    
    # Return state as list (39 elements)
    state = [
        distance_3d, distance_x, distance_y, distance_z,
        own_roll_sin, own_roll_cos, own_pitch_sin, own_pitch_cos, 
        own_heading_sin, own_heading_cos,
        enemy_roll_sin, enemy_roll_cos, enemy_pitch_sin, enemy_pitch_cos, 
        enemy_heading_sin, enemy_heading_cos,
        relative_velocity_x, relative_velocity_y, relative_velocity_z,
        q_radps, p_radps, r_radps,
        current_lock_duration, current_lock_status,
        target_roll, target_pitch, target_throttle,
        los_error_azimuth_cos, los_error_azimuth_sin,
        los_error_elev_cos, los_error_elev_sin,
        prev_roll_pid_dterm, prev_roll_pid_iterm, prev_roll_pid_pterm,
        prev_pitch_pid_pterm, prev_pitch_pid_iterm, prev_pitch_pid_dterm
    ]
    
    return state

def update_attitude(action, target_attitude):
    """Update target attitude based on action deltas"""
    delta_roll = ROLL_ACTION_DELTAS[action[0]]
    delta_pitch = PITCH_ACTION_DELTAS[action[1]]
    delta_throttle = THROTTLE_ACTION_DELTAS[action[2]]
    
    target_attitude[0] = np.clip(
        target_attitude[0] + delta_roll, ROLL_MIN, ROLL_MAX
    )
    target_attitude[1] = np.clip(
        target_attitude[1] + delta_pitch, PITCH_MIN, PITCH_MAX
    )
    target_attitude[2] = np.clip(
        target_attitude[2] + delta_throttle, THROTTLE_MIN, THROTTLE_MAX
    )
    return target_attitude

def apply_action(vehicle, target_attitude):
    """Apply target attitude to vehicle via RC channels"""
    roll_val = target_attitude[0]
    pitch_val = target_attitude[1]
    throttle_val = target_attitude[2]
    
    roll_ref = int(np.interp(roll_val, [ROLL_MIN, ROLL_MAX], [ROLL_CHANNEL_MIN, ROLL_CHANNEL_MAX]))
    alt_ref = int(PITCH_CHANNEL_TRIM)
    airspeed_ref = int(THROTTLE_CHANNEL_TRIM)
    
    vehicle.channels.overrides = {
        '1': roll_ref,
        '2': alt_ref,
        '3': airspeed_ref
    }

class GroundStationClient:
    """Client to communicate with ground station running the ML model"""
    def __init__(self, ground_station_ip, port=5555):
        self.ground_station_ip = ground_station_ip
        self.port = port
        self.socket = None
        self.connect()
    
    def connect(self):
        """Connect to ground station"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(5.0)  # 5 second timeout
                self.socket.connect((self.ground_station_ip, self.port))
                print(f"Connected to ground station at {self.ground_station_ip}:{self.port}")
                return True
            except Exception as e:
                print(f"Connection attempt {retry_count+1} failed: {e}")
                retry_count += 1
                time.sleep(2)
        
        raise ConnectionError(f"Failed to connect to ground station after {max_retries} attempts")
    
    def get_action(self, state):
        """Send state to ground station and receive action"""
        try:
            # Prepare message
            message = {
                'type': 'state',
                'state': state,
                'timestamp': time.time()
            }
            
            # Send state
            json_message = json.dumps(message)
            message_bytes = json_message.encode('utf-8')
            self.socket.sendall(len(message_bytes).to_bytes(4, 'big'))
            self.socket.sendall(message_bytes)
            
            # Receive action
            length_bytes = self.socket.recv(4)
            if not length_bytes:
                raise ConnectionError("Connection closed by ground station")
            
            message_length = int.from_bytes(length_bytes, 'big')
            response_bytes = self.socket.recv(message_length)
            response = json.loads(response_bytes.decode('utf-8'))
            
            if response['type'] == 'action':
                return response['action']
            else:
                raise ValueError(f"Unexpected response type: {response['type']}")
                
        except Exception as e:
            print(f"Error getting action from ground station: {e}")
            # Return safe default action if communication fails
            return [4, 2, 1]  # Center actions (zero deltas)
    
    def close(self):
        """Close connection to ground station"""
        if self.socket:
            try:
                # Send disconnect message
                message = {'type': 'disconnect'}
                json_message = json.dumps(message)
                message_bytes = json_message.encode('utf-8')
                self.socket.sendall(len(message_bytes).to_bytes(4, 'big'))
                self.socket.sendall(message_bytes)
            except:
                pass
            finally:
                self.socket.close()
                print("Disconnected from ground station")

def main():
    # Parameters
    agent_freq = 5  # Hz
    control_freq = 20  # Hz
    agent_period = control_freq // agent_freq  # 4 control steps per agent step
    
    # Parse arguments
    if len(sys.argv) != 4:
        print("Usage: python uav_client.py <flight_time> <ground_station_ip> <mavlink_address>")
        print("Example: python uav_client.py 60 192.168.1.172 127.0.0.1:14550")
        sys.exit(1)
    
    flight_time = float(sys.argv[1])
    ground_station_ip = sys.argv[2]
    machine_address = sys.argv[3]
    
    # Setup
    try:
        # Connect to ground station
        gs_client = GroundStationClient(ground_station_ip, port=5555)
        
        # Connect to vehicle
        vehicle = connect_mavlink(machine_address)
        
        # Setup enemy tracking (optional)
        setup_enemy_tracking(method='mavlink', address='127.0.0.1:14551')
        
        # Initialize PID controllers
        pid_roll = PID(P=0.5, I=0.1, D=0.05)  # Adjust to your values
        pid_pitch = PID(P=0.6, I=0.15, D=0.08)  # Adjust to your values
        
        pid_roll.setWindup(20.0)
        pid_pitch.setWindup(20.0)
        
        # Initialize
        action = [4, 2, 1]  # Center actions
        target_attitude = np.array([0.0, 0.0, 0.5])
        
        # Set flight mode
        vehicle.mode = VehicleMode("FBWB")
        time.sleep(0.5)
        print(f"Vehicle mode set to: {vehicle.mode}")
        
        # Main control loop
        next_time = time.perf_counter()
        dt = 1.0 / control_freq
        total_steps = int(flight_time * control_freq)
        
        print(f"Starting flight for {flight_time} seconds ({total_steps} steps)")
        
        for i in range(total_steps):
            # Agent inference at 5Hz
            if i % agent_period == 0:
                # Get current state
                state = get_state_for_ground_station(vehicle, target_attitude, pid_roll, pid_pitch)
                
                if state is not None:
                    # Get action from ground station
                    action = gs_client.get_action(state)
                    
                    # Update target attitude
                    target_attitude = update_attitude(action, target_attitude)
                    
                    print(f"Step {i}: Action={action}, Target attitude={target_attitude}")
            
            # Update PID controllers
            current_roll = vehicle.attitude.roll * 180 / np.pi
            current_pitch = vehicle.attitude.pitch * 180 / np.pi
            pid_roll.SetPoint = target_attitude[0]
            pid_roll.update(current_roll)
            pid_pitch.SetPoint = target_attitude[1]
            pid_pitch.update(current_pitch)
            
            # Apply action
            apply_action(vehicle, target_attitude)
            
            # Precise timing
            next_time += dt
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        print("Flight completed successfully")
        
        # Set vehicle to AUTO mode after flight
        print("Setting vehicle to AUTO mode...")
        vehicle.channels.overrides = {}  # Clear overrides first
        time.sleep(0.5)  # Give time for overrides to clear
        
        vehicle.mode = VehicleMode("AUTO")
        # Wait for mode change to confirm
        timeout = 5.0
        start_time = time.time()
        while vehicle.mode.name != "AUTO" and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if vehicle.mode.name == "AUTO":
            print(f"Vehicle successfully set to AUTO mode")
        else:
            print(f"Warning: Failed to set AUTO mode, current mode is {vehicle.mode.name}")
            # Try RTL as fallback
            print("Attempting to set RTL mode as fallback...")
            vehicle.mode = VehicleMode("RTL")
            time.sleep(1)
            if vehicle.mode.name == "RTL":
                print("Vehicle set to RTL mode")
            else:
                print(f"Current mode: {vehicle.mode.name}")
        
    except KeyboardInterrupt:
        print("\nFlight interrupted by user")
        # Try to set safe mode on interrupt as well
        if 'vehicle' in locals():
            try:
                print("Setting vehicle to AUTO mode due to interruption...")
                vehicle.channels.overrides = {}
                vehicle.mode = VehicleMode("AUTO")
                time.sleep(1)
                print(f"Vehicle mode: {vehicle.mode.name}")
            except:
                pass
    except Exception as e:
        print(f"Error during flight: {e}")
        import traceback
        traceback.print_exc()
        # Try to set safe mode on error
        if 'vehicle' in locals():
            try:
                print("Setting vehicle to AUTO mode due to error...")
                vehicle.channels.overrides = {}
                vehicle.mode = VehicleMode("AUTO")
                time.sleep(1)
                print(f"Vehicle mode: {vehicle.mode.name}")
            except:
                pass
    finally:
        # Clean up
        if 'gs_client' in locals():
            gs_client.close()
        if 'vehicle' in locals():
            # Final attempt to ensure safe mode
            try:
                if vehicle.mode.name not in ["AUTO", "RTL", "LAND"]:
                    vehicle.channels.overrides = {}
                    vehicle.mode = VehicleMode("AUTO")
                    time.sleep(0.5)
            except:
                pass
            vehicle.close()
            print("Vehicle connection closed")

if __name__ == "__main__":
    main()
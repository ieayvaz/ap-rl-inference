import math
import sys
import time
import numpy as np
from stable_baselines3 import PPO
from dronekit import connect, VehicleMode
from geopy.distance import distance as geo_distance
from attitude_monitor import AttitudeMonitor
from visualizer_multi import DogfightVisualizer

# All params
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
PITCH_CHANNEL_MAX = 1850
PITCH_CHANNEL_MIN = 1250
THROTTLE_CHANNEL_MAX = 1600
THROTTLE_CHANNEL_MIN = 1400

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
    'q_radps': (-1, 1),  # Assuming normalized angular velocities
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

# Functions
def load_model(model_path):
    return PPO.load(model_path)

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
enemy_vehicle = None  # MAVLink connection to enemy drone
enemy_server_url = None  # Server URL for enemy data

def lla_to_ecef(lat, lon, alt):
    """
    Convert LLA (Latitude, Longitude, Altitude) to ECEF (Earth-Centered, Earth-Fixed)
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters (above WGS84 ellipsoid)
    
    Returns:
        numpy array [x, y, z] in ECEF coordinates (meters)
    """
    # WGS84 ellipsoid parameters
    a = 6378137.0  # Semi-major axis (equatorial radius) in meters
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f * f  # First eccentricity squared
    
    # Convert to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    # Calculate prime vertical radius of curvature
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
    
    # Calculate ECEF coordinates
    x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (N * (1 - e2) + alt) * math.sin(lat_rad)
    
    return np.array([x, y, z])

def ecef_to_enu(ecef_point, ref_lat, ref_lon, ref_alt):
    """
    Convert ECEF coordinates to ENU (East, North, Up) relative to a reference point
    
    Args:
        ecef_point: numpy array [x, y, z] in ECEF coordinates
        ref_lat: Reference latitude in degrees
        ref_lon: Reference longitude in degrees
        ref_alt: Reference altitude in meters
    
    Returns:
        numpy array [east, north, up] in ENU coordinates (meters)
    """
    # Get reference point in ECEF
    ref_ecef = lla_to_ecef(ref_lat, ref_lon, ref_alt)
    
    # Calculate relative ECEF coordinates
    delta_ecef = ecef_point - ref_ecef
    
    # Convert to radians
    lat_rad = math.radians(ref_lat)
    lon_rad = math.radians(ref_lon)
    
    # Rotation matrix from ECEF to ENU
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    
    # ENU rotation matrix
    R = np.array([
        [-sin_lon,           cos_lon,          0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon,  cos_lat * sin_lon,  sin_lat]
    ])
    
    # Apply rotation
    enu = R @ delta_ecef
    
    return enu

def lla_to_enu(target_lat, target_lon, target_alt, ref_lat, ref_lon, ref_alt):
    """
    Direct conversion from LLA to ENU coordinates
    
    Args:
        target_lat, target_lon, target_alt: Target position in LLA
        ref_lat, ref_lon, ref_alt: Reference position in LLA
    
    Returns:
        numpy array [east, north, up] in meters
    """
    target_ecef = lla_to_ecef(target_lat, target_lon, target_alt)
    return ecef_to_enu(target_ecef, ref_lat, ref_lon, ref_alt)

def setup_enemy_tracking(method='mavlink', address=None):
    """
    Setup enemy tracking method
    Args:
        method: 'mavlink' or 'server'
        address: MAVLink address or server URL
    """
    global enemy_vehicle, enemy_server_url
    
    if method == 'mavlink' and address:
        enemy_vehicle = connect(address, wait_ready=True)
        print(f"Connected to enemy vehicle at {address}")
    elif method == 'server' and address:
        enemy_server_url = address
        print(f"Will fetch enemy data from server: {address}")

def get_enemy_data_from_server():
    """
    Get enemy aircraft data from a server
    Returns dict with: lat, lon, alt, roll, pitch, heading, velocity
    """
    # TODO: Implement server communication
    # This would typically involve:
    # 1. Making HTTP/TCP request to server
    # 2. Parsing JSON/binary response
    # 3. Returning structured data
    raise NotImplementedError("Server method not yet implemented")

def get_enemy_data_from_mavlink():
    """
    Get enemy aircraft data from MAVLink connection
    Returns dict with: lat, lon, alt, roll, pitch, heading, velocity
    """
    if enemy_vehicle is None:
        return None
    
    return {
        'lat': enemy_vehicle.location.global_frame.lat,
        'lon': enemy_vehicle.location.global_frame.lon,
        'alt': enemy_vehicle.location.global_frame.alt,
        'roll': enemy_vehicle.attitude.roll,
        'pitch': enemy_vehicle.attitude.pitch,
        'heading': enemy_vehicle.heading,
        'velocity': enemy_vehicle.velocity  # [North, East, Down] in m/s
    }

def calculate_los_angles_and_errors(enu_vector, own_heading, own_pitch):
    """
    Calculate Line of Sight (LOS) angles and errors matching your exact implementation
    
    Args:
        enu_vector: numpy array [east, north, up] in meters
        own_heading: aircraft heading in degrees (0-360)
        own_pitch: aircraft pitch in radians
    
    Returns:
        tuple: (los_azimuth, los_elevation, azimuth_error, elevation_error)
               angles in degrees for azimuth, radians for elevation
    """
    # Extract ENU components matching your get_distance_v()
    # distance vector is [east, north, up]
    east = enu_vector[0]   # dist[0] in your code
    north = enu_vector[1]  # dist[1] in your code  
    up = enu_vector[2]     # dist[2] in your code
    
    # Calculate LOS azimuth (matching get_3d_los_azimuth)
    angle_rad = math.atan2(north, east)  # atan2(dist[1], dist[0])
    angle_deg = math.degrees(angle_rad)
    los_azimuth = (90 - angle_deg + 360) % 360  # Convert to compass heading
    
    # Calculate LOS elevation (matching get_3d_los_elevation)
    horizontal_dist = math.sqrt(east**2 + north**2)
    elevation_rad = math.atan2(up, horizontal_dist)
    los_elevation = math.degrees(elevation_rad)
    
    # Calculate azimuth error (matching get_3d_los_azimuth_error)
    azimuth_error = los_azimuth - own_heading
    # Normalize to [-180, 180]
    if azimuth_error > 180:
        azimuth_error -= 360
    elif azimuth_error < -180:
        azimuth_error += 360
    
    # Calculate elevation error
    own_pitch_deg = math.degrees(own_pitch)
    elevation_error = los_elevation - own_pitch_deg
    
    return los_azimuth, los_elevation, azimuth_error, elevation_error

def check_lock_status(distance_3d, azimuth_error, elevation_error):
    """
    Check if locked onto target using your exact curriculum parameters
    
    Args:
        distance_3d: distance to target in meters
        azimuth_error: azimuth error in degrees
        elevation_error: elevation error in degrees
    
    Returns:
        bool: True if locked, False otherwise
    """
    # Distance check: must be between 10 and 50 meters
    if distance_3d > 50 or distance_3d < 10:
        return False
    
    # Azimuth error check: must be within ±25 degrees
    if abs(azimuth_error) > 25:
        return False
    
    # Elevation error check: must be within ±15 degrees
    if abs(elevation_error) > 15:
        return False
    
    return True

# Store lock duration across calls
lock_start_time = None
current_lock_duration = 0.0

def normalize_state(state):
    """
    Normalize state vector to [-1, 1] range using training bounds
    
    Args:
        state: numpy array of raw state values (39 elements)
    
    Returns:
        numpy array of normalized state values
    """
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
        # Formula: 2 * (x - min) / (max - min) - 1
        if max_val > min_val:
            normalized_value = 2 * (clipped_value - min_val) / (max_val - min_val) - 1
        else:
            normalized_value = 0  # Handle edge case where min == max
        
        normalized_state[i] = normalized_value
    
    return normalized_state.astype(np.float32)

def get_state(vehicle, target_attitude, pid_roll, pid_pitch):
    """
    Extract state observation for the RL model
    
    Args:
        vehicle: DroneKit vehicle object (own aircraft)
        target_attitude: [roll, pitch, throttle] target values
        pid_roll: PID controller for roll
        pid_pitch: PID controller for pitch
    
    Returns:
        numpy array of state variables (39 elements)
    """
    global lock_start_time, current_lock_duration
    
    # Get own aircraft data
    own_lat = vehicle.location.global_frame.lat
    own_lon = vehicle.location.global_frame.lon
    own_alt = vehicle.location.global_frame.alt
    own_roll = vehicle.attitude.roll
    own_pitch = vehicle.attitude.pitch
    own_heading = vehicle.heading
    own_velocity = vehicle.velocity  # [North, East, Down] m/s
    
    # Get enemy aircraft data
    if enemy_vehicle is not None:
        enemy_data = get_enemy_data_from_mavlink()
    elif enemy_server_url is not None:
        enemy_data = get_enemy_data_from_server()
    else:
        # Default values if no enemy tracking configured
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
    
    # Calculate distances using ENU coordinates
    own_pos = {'lat': own_lat, 'lon': own_lon, 'alt': own_alt}
    enemy_pos = {'lat': enemy_data['lat'], 'lon': enemy_data['lon'], 'alt': enemy_data['alt']}
    
    # Get enemy position in ENU coordinates relative to own position
    enu_enemy = lla_to_enu(
        enemy_data['lat'], enemy_data['lon'], enemy_data['alt'],
        own_lat, own_lon, own_alt
    )
    
    # ENU components: East, North, Up
    distance_east = enu_enemy[0]
    distance_north = enu_enemy[1]
    distance_up = enu_enemy[2]
    
    # For state vector, use ENU ordering: x=East, y=North, z=Up
    distance_x = distance_east   # East component
    distance_y = distance_north   # North component
    distance_z = distance_up      # Up component (altitude difference)
    
    # 3D Euclidean distance
    distance_3d = math.sqrt(distance_x**2 + distance_y**2 + distance_z**2)
    
    # Trigonometric features for attitudes
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
    relative_velocity_x = enemy_data['velocity'][0] - own_velocity[0]  # North
    relative_velocity_y = enemy_data['velocity'][1] - own_velocity[1]  # East
    relative_velocity_z = enemy_data['velocity'][2] - own_velocity[2]  # Down
    
    # Angular velocities (rad/s)
    # Using gyro data if available, otherwise estimate from attitude changes
    try:
        p_radps = vehicle.raw_imu.xgyro / 1000.0  # Convert from mrad/s to rad/s
        q_radps = vehicle.raw_imu.ygyro / 1000.0
        r_radps = vehicle.raw_imu.zgyro / 1000.0
    except:
        # Fallback: use zero if IMU data not available
        p_radps = 0.0
        q_radps = 0.0
        r_radps = 0.0
    
    # Calculate LOS angles and errors using your exact method
    los_azimuth, los_elevation, azimuth_error_deg, elevation_error_deg = \
        calculate_los_angles_and_errors(enu_enemy, own_heading, own_pitch)
    
    # Convert errors to radians for state vector (keeping degrees for lock check)
    los_azimuth_error_rad = math.radians(azimuth_error_deg)
    los_elevation_error_rad = math.radians(elevation_error_deg)
    
    los_error_azimuth_cos = math.cos(los_azimuth_error_rad)
    los_error_azimuth_sin = math.sin(los_azimuth_error_rad)
    los_error_elev_cos = math.cos(los_elevation_error_rad)
    los_error_elev_sin = math.sin(los_elevation_error_rad)
    
    # Lock status using your exact conditions
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
    
    # PID terms from controllers
    prev_roll_pid_pterm = pid_roll.PTerm
    prev_roll_pid_iterm = pid_roll.ITerm
    prev_roll_pid_dterm = pid_roll.DTerm
    prev_pitch_pid_pterm = pid_pitch.PTerm
    prev_pitch_pid_iterm = pid_pitch.ITerm
    prev_pitch_pid_dterm = pid_pitch.DTerm
    
    # Assemble state vector (39 elements)
    state = np.array([
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
    ], dtype=np.float32)
    
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
    """Apply target attitude to vehicle via RC channel reference values (FBWA mode)"""
    roll_val = target_attitude[0]  # degrees
    pitch_val = target_attitude[1]  # degrees
    throttle_val = target_attitude[2]  # [0.4, 0.6]
    
    # Map from desired angles/throttle to RC channel reference values
    # In FBWA mode, these are interpreted as angle/throttle references, not raw PWM
    roll_ref = int(np.interp(roll_val, [ROLL_MIN, ROLL_MAX], [ROLL_CHANNEL_MIN, ROLL_CHANNEL_MAX]))
    pitch_ref = int(np.interp(pitch_val, [PITCH_MIN, PITCH_MAX], [PITCH_CHANNEL_MIN, PITCH_CHANNEL_MAX]))
    throttle_ref = int(np.interp(throttle_val, [THROTTLE_MIN, THROTTLE_MAX], [THROTTLE_CHANNEL_MIN, THROTTLE_CHANNEL_MAX]))
    
    vehicle.channels.overrides = {
        '1': roll_ref,
        '2': pitch_ref,
        '3': throttle_ref,
    }

def main():
    # Parameters
    agent_freq = 5  # Hz
    control_freq = 20  # Hz
    agent_period = control_freq // agent_freq  # 4 control steps per agent step

    visualize = True
    
    # Parse arguments
    if len(sys.argv) != 4:
        print("Usage: python script.py <flight_time> <model_path> <machine_address>")
        sys.exit(1)
    
    flight_time = float(sys.argv[1])
    model_path = sys.argv[2]
    machine_address = sys.argv[3]
    
    # Setup
    try:
        model = load_model(model_path)
        vehicle = connect_mavlink(machine_address)
        setup_enemy_tracking(method='mavlink', address='127.0.0.1:14551')

        visualizer = None
        attitude_monitor = None
        if visualize:
            visualizer = DogfightVisualizer()
            attitude_monitor = AttitudeMonitor()

        # Initialize PID controllers for roll and pitch
        # IMPORTANT: Replace these with your actual ArduPilot PID coefficients
        pid_roll = PID(P=0.5, I=0.1, D=0.05)  # Adjust to your values
        pid_pitch = PID(P=0.6, I=0.15, D=0.08)  # Adjust to your values
        
        # Set windup guards
        pid_roll.setWindup(20.0)
        pid_pitch.setWindup(20.0)
        
        # Initialize
        action = np.array([4, 2, 1])  # Center actions (indices for zero deltas)
        target_attitude = np.array([0.0, 0.0, 0.5])  # Initial roll, pitch, throttle
        
        # Set flight mode
        vehicle.mode = VehicleMode("FBWA")
        time.sleep(0.5)
        print(f"Vehicle mode set to: {vehicle.mode}")
        
        # Get initial observation
        obs = get_state(vehicle, target_attitude, pid_roll, pid_pitch)
        
        # Main control loop
        next_time = time.perf_counter()
        dt = 1.0 / control_freq
        total_steps = int(flight_time * control_freq)
        
        print(f"Starting flight for {flight_time} seconds ({total_steps} steps)")
        
        for i in range(total_steps):
            # Agent inference at 5Hz
            if i % agent_period == 0:
                obs = get_state(vehicle, target_attitude, pid_roll, pid_pitch)

                if visualize:
                    visualizer.update_state(obs)
                    visualizer.update_plot()
                    attitude_monitor.update_state(obs)

                obs_norm = normalize_state(obs)
                if obs_norm is not None:  # Only predict if we have valid state
                    action, _ = model.predict(obs_norm, deterministic=False) 
                    target_attitude = update_attitude(action, target_attitude)
                    print(f"Step {i}: Action={action}, Target attitude={target_attitude}")
            
            # Update PID controllers
            current_roll = vehicle.attitude.roll * 180 / np.pi  # Convert to degrees
            current_pitch = vehicle.attitude.pitch * 180 / np.pi
            pid_roll.SetPoint = target_attitude[0]
            pid_roll.update(current_roll)
            pid_pitch.SetPoint = target_attitude[1]
            pid_pitch.update(current_pitch)

            apply_action(vehicle, target_attitude)
            
            # Precise timing
            next_time += dt
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        print("Flight completed successfully")
        
    except KeyboardInterrupt:
        print("\nFlight interrupted by user")
    except Exception as e:
        print(f"Error during flight: {e}")
    finally:
        # Clean up
        if 'vehicle' in locals():
            vehicle.channels.overrides = {}  # Clear overrides
            vehicle.close()
            print("Vehicle connection closed")

if __name__ == "__main__":
    main()

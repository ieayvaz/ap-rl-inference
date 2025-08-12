import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.animation as animation
from typing import Dict, List, Tuple, Optional
import math


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class DogfightVisualizer:
    """3D visualizer for dogfight scenarios - Deployment version"""
    
    def __init__(self, max_history_points: int = 100, update_interval: int = 100):
        """
        Initialize the 3D visualizer
        
        Args:
            max_history_points: Maximum number of trajectory points to keep
            update_interval: Animation update interval in milliseconds
        """
        self.max_history_points = max_history_points
        self.update_interval = update_interval
        
        # Data storage
        self.own_trajectory = []
        self.enemy_trajectory = []
        self.own_orientation_history = []
        self.enemy_orientation_history = []
        self.lock_status_history = []
        self.distance_history = []
        
        # Matplotlib setup
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Aircraft markers and trajectories
        self.own_aircraft_marker = None
        self.enemy_aircraft_marker = None
        self.own_trajectory_line = None
        self.enemy_trajectory_line = None
        self.own_orientation_arrow = None
        self.enemy_orientation_arrow = None
        
        # Information text
        self.info_text = None
        
        # Lock indicator
        self.lock_indicator = None
        
        # Animation object
        self.animation = None
        
        # Setup the plot
        self._setup_plot()
        
    def _setup_plot(self):
        """Initialize the plot elements"""
        # Set labels and title
        self.ax.set_xlabel('East (m)')
        self.ax.set_ylabel('North (m)')
        self.ax.set_zlabel('Up (m)')
        self.ax.set_title('Dogfight 3D Visualization - Deployment')
        
        # Set initial view
        self.ax.view_init(elev=20, azim=45)
        
        # Initialize empty trajectory lines
        self.own_trajectory_line, = self.ax.plot([], [], [], 'b-', linewidth=2, alpha=0.7, label='Own Aircraft')
        self.enemy_trajectory_line, = self.ax.plot([], [], [], 'r-', linewidth=2, alpha=0.7, label='Enemy Aircraft')
        
        # Initialize aircraft markers
        self.own_aircraft_marker, = self.ax.plot([], [], [], 'bo', markersize=10, label='Own Position')
        self.enemy_aircraft_marker, = self.ax.plot([], [], [], 'rs', markersize=10, label='Enemy Position')
        
        # Initialize lock indicator line (green when locked)
        self.lock_indicator, = self.ax.plot([], [], [], 'g--', linewidth=2, alpha=0.8)
        
        # Add legend
        self.ax.legend(loc='upper right')
        
        # Add info text
        self.info_text = self.fig.text(0.02, 0.98, '', transform=self.fig.transFigure, 
                                      verticalalignment='top', fontsize=10, 
                                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Set grid
        self.ax.grid(True, alpha=0.3)
    
    def update_state(self, state: np.ndarray, origin: Tuple[float, float, float] = (0, 0, 0)):
        """
        Update visualization from state vector
        
        Args:
            state: Raw state vector from get_state() function (39 elements)
            origin: Reference origin for visualization (default: 0,0,0)
        
        State vector indices:
        0: distance_3d
        1-3: distance_x (east), distance_y (north), distance_z (up)
        4-9: own aircraft sin/cos values
        10-15: enemy aircraft sin/cos values
        16-18: relative velocities
        19-21: angular velocities (q, p, r)
        22: lock duration
        23: lock status
        24-26: target roll, pitch, throttle
        27-30: LOS error sin/cos values
        31-36: PID terms
        """
        
        # Extract relevant information from state
        distance_3d = state[0]
        distance_east = state[1]   # x component in ENU
        distance_north = state[2]  # y component in ENU
        distance_up = state[3]      # z component in ENU
        
        # Reconstruct orientations from sin/cos values
        own_roll = math.atan2(state[4], state[5])      # atan2(sin, cos)
        own_pitch = math.atan2(state[6], state[7])
        own_heading = math.atan2(state[8], state[9])
        
        enemy_roll = math.atan2(state[10], state[11])
        enemy_pitch = math.atan2(state[12], state[13])
        enemy_heading = math.atan2(state[14], state[15])
        
        # Lock status
        lock_duration = state[22]
        lock_status = state[23]
        
        # Target attitude (for display)
        target_roll = state[24]
        target_pitch = state[25]
        target_throttle = state[26]
        
        # Calculate positions
        # Own aircraft at origin
        own_pos = (origin[0], origin[1], origin[2])
        
        # Enemy aircraft relative to own
        enemy_pos = (
            origin[0] + distance_east,
            origin[1] + distance_north,
            origin[2] + distance_up
        )
        
        # Store data
        self.own_trajectory.append(own_pos)
        self.enemy_trajectory.append(enemy_pos)
        self.own_orientation_history.append((own_roll, own_pitch, own_heading))
        self.enemy_orientation_history.append((enemy_roll, enemy_pitch, enemy_heading))
        self.lock_status_history.append(lock_status)
        self.distance_history.append(distance_3d)
        
        # Limit history length
        if len(self.own_trajectory) > self.max_history_points:
            self.own_trajectory.pop(0)
            self.enemy_trajectory.pop(0)
            self.own_orientation_history.pop(0)
            self.enemy_orientation_history.pop(0)
            self.lock_status_history.pop(0)
            self.distance_history.pop(0)
        
        # Store additional info for display
        self.current_info = {
            'distance': distance_3d,
            'lock_status': lock_status,
            'lock_duration': lock_duration,
            'target_roll': target_roll,
            'target_pitch': target_pitch,
            'target_throttle': target_throttle,
            'relative_bearing': math.degrees(math.atan2(distance_east, distance_north)),
            'relative_elevation': math.degrees(math.atan2(distance_up, 
                                                          math.sqrt(distance_east**2 + distance_north**2)))
        }
    
    def _get_aircraft_orientation_vector(self, orientation: Tuple[float, float, float],
                                        length: float = 50.0) -> np.ndarray:
        """
        Calculate aircraft nose direction vector from orientation
        
        Args:
            orientation: (roll, pitch, heading) in radians
            length: Length of the orientation vector
        
        Returns:
            3D vector pointing in aircraft's forward direction
        """
        roll, pitch, heading = orientation
        
        # Convert heading to ENU frame (0° = North, 90° = East)
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)
        
        # Forward vector in ENU
        forward = np.array([
            cos_p * sin_h,  # East component
            cos_p * cos_h,  # North component
            sin_p           # Up component
        ]) * length
        
        return forward
    
    def update_plot(self):
        """Update the 3D plot with current data"""
        if not self.own_trajectory or not self.enemy_trajectory:
            return
        
        # Update trajectory lines
        if len(self.own_trajectory) > 1:
            own_traj = np.array(self.own_trajectory)
            self.own_trajectory_line.set_data_3d(own_traj[:, 0], own_traj[:, 1], own_traj[:, 2])
        
        if len(self.enemy_trajectory) > 1:
            enemy_traj = np.array(self.enemy_trajectory)
            self.enemy_trajectory_line.set_data_3d(enemy_traj[:, 0], enemy_traj[:, 1], enemy_traj[:, 2])
        
        # Update aircraft markers
        own_pos = self.own_trajectory[-1]
        enemy_pos = self.enemy_trajectory[-1]
        
        self.own_aircraft_marker.set_data_3d([own_pos[0]], [own_pos[1]], [own_pos[2]])
        self.enemy_aircraft_marker.set_data_3d([enemy_pos[0]], [enemy_pos[1]], [enemy_pos[2]])
        
        # Update lock indicator
        if self.lock_status_history[-1] > 0.5:
            # Draw line between aircraft when locked
            self.lock_indicator.set_data_3d([own_pos[0], enemy_pos[0]], 
                                           [own_pos[1], enemy_pos[1]], 
                                           [own_pos[2], enemy_pos[2]])
            self.lock_indicator.set_color('g')
            self.lock_indicator.set_linestyle('-')
            self.lock_indicator.set_alpha(0.8)
        else:
            # Faint line when not locked
            self.lock_indicator.set_data_3d([own_pos[0], enemy_pos[0]], 
                                           [own_pos[1], enemy_pos[1]], 
                                           [own_pos[2], enemy_pos[2]])
            self.lock_indicator.set_color('gray')
            self.lock_indicator.set_linestyle(':')
            self.lock_indicator.set_alpha(0.3)
        
        # Update orientation arrows
        if self.own_orientation_history and self.enemy_orientation_history:
            self._update_orientation_arrows()
        
        # Update plot limits
        self._update_plot_limits()
        
        # Update info text
        if hasattr(self, 'current_info'):
            lock_str = "LOCKED" if self.current_info['lock_status'] > 0.5 else "NO LOCK"
            lock_color = "🟢" if self.current_info['lock_status'] > 0.5 else "🔴"
            
            info_str = f"{lock_color} {lock_str}\n"
            info_str += f"Distance: {self.current_info['distance']:.1f}m\n"
            info_str += f"Lock Duration: {self.current_info['lock_duration']:.1f}s\n"
            info_str += f"Rel Bearing: {self.current_info['relative_bearing']:.1f}°\n"
            info_str += f"Rel Elevation: {self.current_info['relative_elevation']:.1f}°\n"
            info_str += f"─────────────\n"
            info_str += f"Target Roll: {self.current_info['target_roll']:.1f}°\n"
            info_str += f"Target Pitch: {self.current_info['target_pitch']:.1f}°\n"
            info_str += f"Target Throttle: {self.current_info['target_throttle']:.2f}\n"
            info_str += f"Points: {len(self.own_trajectory)}"
            
            self.info_text.set_text(info_str)
        
        # Color trajectory based on lock status
        if len(self.lock_status_history) > 1:
            # Create segments colored by lock status
            locked_segments = []
            unlocked_segments = []
            
            for i in range(1, len(self.own_trajectory)):
                if self.lock_status_history[i] > 0.5:
                    locked_segments.append(i)
                else:
                    unlocked_segments.append(i)
        
        # Redraw
        plt.pause(0.00001)
        self.fig.canvas.draw()
    
    def _update_orientation_arrows(self):
        """Update aircraft orientation arrows"""
        if not (self.own_orientation_history and self.enemy_orientation_history):
            return
        
        own_pos = np.array(self.own_trajectory[-1])
        enemy_pos = np.array(self.enemy_trajectory[-1])
        
        # Remove old arrows
        if self.own_orientation_arrow:
            self.own_orientation_arrow.remove()
        if self.enemy_orientation_arrow:
            self.enemy_orientation_arrow.remove()
        
        # Calculate orientation vectors
        own_forward = self._get_aircraft_orientation_vector(self.own_orientation_history[-1])
        enemy_forward = self._get_aircraft_orientation_vector(self.enemy_orientation_history[-1])
        
        # Create new arrows
        self.own_orientation_arrow = Arrow3D([own_pos[0], own_pos[0] + own_forward[0]],
                                            [own_pos[1], own_pos[1] + own_forward[1]],
                                            [own_pos[2], own_pos[2] + own_forward[2]],
                                            mutation_scale=20, lw=2, arrowstyle="-|>", color="blue")
        
        self.enemy_orientation_arrow = Arrow3D([enemy_pos[0], enemy_pos[0] + enemy_forward[0]],
                                              [enemy_pos[1], enemy_pos[1] + enemy_forward[1]],
                                              [enemy_pos[2], enemy_pos[2] + enemy_forward[2]],
                                              mutation_scale=20, lw=2, arrowstyle="-|>", color="red")
        
        self.ax.add_artist(self.own_orientation_arrow)
        self.ax.add_artist(self.enemy_orientation_arrow)
    
    def _update_plot_limits(self):
        """Update plot limits based on current data"""
        if not self.own_trajectory or not self.enemy_trajectory:
            return
        
        all_positions = np.array(self.own_trajectory + self.enemy_trajectory)
        
        # Add some padding
        padding = 100  # meters
        
        x_min, x_max = all_positions[:, 0].min() - padding, all_positions[:, 0].max() + padding
        y_min, y_max = all_positions[:, 1].min() - padding, all_positions[:, 1].max() + padding
        z_min, z_max = all_positions[:, 2].min() - padding, all_positions[:, 2].max() + padding
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_zlim(max(0, z_min), z_max)  # Keep altitude above 0
    
    def add_engagement_zone(self, center: Tuple[float, float, float], inner_radius: float = 10, outer_radius: float = 50):
        """Add engagement zone visualization (10-50m lock zone)"""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        # Outer boundary (50m)
        x_outer = center[0] + outer_radius * np.outer(np.cos(u), np.sin(v))
        y_outer = center[1] + outer_radius * np.outer(np.sin(u), np.sin(v))
        z_outer = center[2] + outer_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Inner boundary (10m)
        x_inner = center[0] + inner_radius * np.outer(np.cos(u), np.sin(v))
        y_inner = center[1] + inner_radius * np.outer(np.sin(u), np.sin(v))
        z_inner = center[2] + inner_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.ax.plot_wireframe(x_outer, y_outer, z_outer, alpha=0.2, color='green', label='Max Lock Range')
        self.ax.plot_wireframe(x_inner, y_inner, z_inner, alpha=0.2, color='red', label='Min Lock Range')
    
    def start_animation(self, update_function=None):
        """Start real-time animation"""
        def animate(frame):
            if update_function:
                update_function()
            self.update_plot()
            return []
        
        self.animation = animation.FuncAnimation(self.fig, animate, interval=self.update_interval, 
                                                blit=False, repeat=True)
    
    def stop_animation(self):
        """Stop the animation"""
        if self.animation:
            self.animation.event_source.stop()
    
    def save_animation(self, filename: str, duration: int = 10):
        """Save animation as gif or mp4"""
        if self.animation:
            self.animation.save(filename, writer='pillow' if filename.endswith('.gif') else 'ffmpeg',
                               fps=1000//self.update_interval)
    
    def clear_trajectories(self):
        """Clear trajectory history"""
        self.own_trajectory.clear()
        self.enemy_trajectory.clear()
        self.own_orientation_history.clear()
        self.enemy_orientation_history.clear()
        self.lock_status_history.clear()
        self.distance_history.clear()
    
    def show(self):
        """Display the plot"""
        plt.show()
    
    def close(self):
        """Close the plot"""
        plt.close(self.fig)
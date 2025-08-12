import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import math


class AttitudeMonitor:
    """Standalone attitude monitor window for deployment system"""
    
    def __init__(self):
        """Initialize the attitude monitor window"""
        # Create figure
        self.fig = plt.figure(figsize=(10, 6))
        self.fig.suptitle('Attitude Control Monitoring - Deployment', fontsize=16, fontweight='bold')
        
        # Current values
        self.current_target_roll = 0.0
        self.current_target_pitch = 0.0
        self.current_target_throttle = 0.5
        self.current_actual_roll = 0.0
        self.current_actual_pitch = 0.0
        
        # PID terms
        self.roll_pid_p = 0.0
        self.roll_pid_i = 0.0
        self.roll_pid_d = 0.0
        self.pitch_pid_p = 0.0
        self.pitch_pid_i = 0.0
        self.pitch_pid_d = 0.0
        
        # History for plotting
        self.history_length = 100
        self.time_history = []
        self.target_roll_history = []
        self.target_pitch_history = []
        self.actual_roll_history = []
        self.actual_pitch_history = []
        self.throttle_history = []
        
        # Setup UI elements
        self._setup_sliders()
        self._setup_plots()
        self._setup_pid_display()
        
        # Show the window
        plt.show(block=False)
        
    def _setup_sliders(self):
        """Setup the attitude sliders"""
        # Define layout parameters
        slider_height = 0.03
        slider_width = 0.55
        slider_left = 0.15
        plot_bottom = 0.55
        
        # Target Roll
        ax_target_roll = plt.axes([slider_left, plot_bottom - 0.08, slider_width, slider_height])
        self.slider_target_roll = Slider(
            ax_target_roll, 'Target Roll', -60, 60, valinit=0,
            color='green', valfmt='%0.1f°'
        )
        self.slider_target_roll.set_active(False)
        
        # Target Pitch
        ax_target_pitch = plt.axes([slider_left, plot_bottom - 0.13, slider_width, slider_height])
        self.slider_target_pitch = Slider(
            ax_target_pitch, 'Target Pitch', -25, 25, valinit=0,
            color='green', valfmt='%0.1f°'
        )
        self.slider_target_pitch.set_active(False)
        
        # Actual Roll
        ax_actual_roll = plt.axes([slider_left, plot_bottom - 0.20, slider_width, slider_height])
        self.slider_actual_roll = Slider(
            ax_actual_roll, 'Actual Roll', -60, 60, valinit=0,
            color='blue', valfmt='%0.1f°'
        )
        self.slider_actual_roll.set_active(False)
        
        # Actual Pitch
        ax_actual_pitch = plt.axes([slider_left, plot_bottom - 0.25, slider_width, slider_height])
        self.slider_actual_pitch = Slider(
            ax_actual_pitch, 'Actual Pitch', -25, 25, valinit=0,
            color='blue', valfmt='%0.1f°'
        )
        self.slider_actual_pitch.set_active(False)
        
        # Throttle slider
        ax_throttle = plt.axes([slider_left, plot_bottom - 0.32, slider_width, slider_height])
        self.slider_throttle = Slider(
            ax_throttle, 'Throttle', 0.0, 1.0, valinit=0.5,
            color='orange', valfmt='%0.2f'
        )
        self.slider_throttle.set_active(False)
        
        # Add labels
        self.fig.text(0.05, plot_bottom - 0.09, 'RL Agent\nCommand', 
                     verticalalignment='center', fontsize=10, color='green', weight='bold')
        self.fig.text(0.05, plot_bottom - 0.21, 'Aircraft\nResponse', 
                     verticalalignment='center', fontsize=10, color='blue', weight='bold')
        
        # Error display
        self.error_text = self.fig.text(0.75, plot_bottom - 0.15, '', 
                                       verticalalignment='center', fontsize=10,
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        
    def _setup_plots(self):
        """Setup time history plots"""
        # Create subplots for roll and pitch history
        self.ax_roll = plt.axes([0.1, 0.65, 0.60, 0.13])
        self.ax_pitch = plt.axes([0.1, 0.79, 0.60, 0.13])
        
        # Initialize plot lines
        self.line_target_roll, = self.ax_roll.plot([], [], 'g-', label='Target', linewidth=2)
        self.line_actual_roll, = self.ax_roll.plot([], [], 'b-', label='Actual', linewidth=2)
        self.ax_roll.set_ylabel('Roll (°)')
        self.ax_roll.set_ylim(-60, 60)
        self.ax_roll.grid(True, alpha=0.3)
        self.ax_roll.legend(loc='upper right', fontsize=8)
        
        self.line_target_pitch, = self.ax_pitch.plot([], [], 'g-', label='Target', linewidth=2)
        self.line_actual_pitch, = self.ax_pitch.plot([], [], 'b-', label='Actual', linewidth=2)
        self.ax_pitch.set_ylabel('Pitch (°)')
        self.ax_pitch.set_ylim(-25, 25)
        self.ax_pitch.grid(True, alpha=0.3)
        self.ax_pitch.legend(loc='upper right', fontsize=8)
        self.ax_pitch.set_xlabel('Time Steps')
        
    def _setup_pid_display(self):
        """Setup PID terms display"""
        # PID display area
        pid_left = 0.73
        pid_top = 0.92
        
        self.pid_text_title = self.fig.text(pid_left, pid_top, 'PID Terms', 
                                           fontsize=11, weight='bold')
        
        self.pid_text_roll = self.fig.text(pid_left, pid_top - 0.08, '', 
                                          fontsize=9, family='monospace',
                                          bbox=dict(boxstyle="round,pad=0.3", 
                                                   facecolor="lightblue", alpha=0.7))
        
        self.pid_text_pitch = self.fig.text(pid_left, pid_top - 0.18, '', 
                                           fontsize=9, family='monospace',
                                           bbox=dict(boxstyle="round,pad=0.3", 
                                                    facecolor="lightgreen", alpha=0.7))
    
    def update_state(self, state: np.ndarray):
        """Update the monitor from state vector
        
        Args:
            state: Raw state vector from get_state() function (39 elements)
            vehicle: Optional DroneKit vehicle object for actual attitude
        
        State vector indices:
        4-5: own_roll_sin, own_roll_cos
        6-7: own_pitch_sin, own_pitch_cos
        24: target_roll
        25: target_pitch
        26: target_throttle
        31-33: roll PID terms (D, I, P)
        34-36: pitch PID terms (P, I, D)
        """
        
        # Extract target attitudes (already in degrees)
        self.current_target_roll = state[24]
        self.current_target_pitch = state[25]
        self.current_target_throttle = state[26]
        
        # Get actual attitudes
        actual_roll_rad = math.atan2(state[4], state[5])  # atan2(sin, cos)
        actual_pitch_rad = math.atan2(state[6], state[7])
        self.current_actual_roll = math.degrees(actual_roll_rad)
        self.current_actual_pitch = math.degrees(actual_pitch_rad)
        
        # Extract PID terms
        self.roll_pid_d = state[31]
        self.roll_pid_i = state[32]
        self.roll_pid_p = state[33]
        self.pitch_pid_p = state[34]
        self.pitch_pid_i = state[35]
        self.pitch_pid_d = state[36]
        
        # Update displays
        self._update_displays()
        
    def update(self, target_roll_deg, target_pitch_deg, target_throttle, 
               actual_roll_rad, actual_pitch_rad,
               roll_pid_p=0, roll_pid_i=0, roll_pid_d=0,
               pitch_pid_p=0, pitch_pid_i=0, pitch_pid_d=0):
        """Direct update with individual values
        
        Args:
            target_roll_deg: Target roll in degrees
            target_pitch_deg: Target pitch in degrees
            target_throttle: Target throttle (0-1)
            actual_roll_rad: Actual roll in radians
            actual_pitch_rad: Actual pitch in radians
            roll_pid_p/i/d: Roll PID terms
            pitch_pid_p/i/d: Pitch PID terms
        """
        # Store values
        self.current_target_roll = target_roll_deg
        self.current_target_pitch = target_pitch_deg
        self.current_target_throttle = target_throttle
        self.current_actual_roll = math.degrees(actual_roll_rad)
        self.current_actual_pitch = math.degrees(actual_pitch_rad)
        
        # Store PID terms
        self.roll_pid_p = roll_pid_p
        self.roll_pid_i = roll_pid_i
        self.roll_pid_d = roll_pid_d
        self.pitch_pid_p = pitch_pid_p
        self.pitch_pid_i = pitch_pid_i
        self.pitch_pid_d = pitch_pid_d
        
        # Update displays
        self._update_displays()
        
    def _update_displays(self):
        """Update all display elements"""
        # Update sliders
        self.slider_target_roll.set_val(self.current_target_roll)
        self.slider_target_pitch.set_val(self.current_target_pitch)
        self.slider_throttle.set_val(self.current_target_throttle)
        self.slider_actual_roll.set_val(self.current_actual_roll)
        self.slider_actual_pitch.set_val(self.current_actual_pitch)
        
        # Calculate errors
        roll_error = abs(self.current_target_roll - self.current_actual_roll)
        pitch_error = abs(self.current_target_pitch - self.current_actual_pitch)
        
        # Update error display
        error_text = f"Roll Error: {roll_error:.1f}°\nPitch Error: {pitch_error:.1f}°"
        self.error_text.set_text(error_text)
        
        # Update PID display
        roll_pid_text = f"Roll PID:\nP: {self.roll_pid_p:+6.2f}\nI: {self.roll_pid_i:+6.2f}\nD: {self.roll_pid_d:+6.2f}"
        pitch_pid_text = f"Pitch PID:\nP: {self.pitch_pid_p:+6.2f}\nI: {self.pitch_pid_i:+6.2f}\nD: {self.pitch_pid_d:+6.2f}"
        self.pid_text_roll.set_text(roll_pid_text)
        self.pid_text_pitch.set_text(pitch_pid_text)
        
        # Color code based on error
        self._update_slider_colors(roll_error, pitch_error)
        
        # Update history
        self._update_history()
        
        # Refresh display
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
    def _update_slider_colors(self, roll_error, pitch_error):
        """Update slider colors based on tracking error"""
        # Roll error coloring
        if roll_error > 15:
            self.slider_actual_roll.poly.set_facecolor('red')
        elif roll_error > 7:
            self.slider_actual_roll.poly.set_facecolor('orange')
        else:
            self.slider_actual_roll.poly.set_facecolor('blue')
            
        # Pitch error coloring
        if pitch_error > 10:
            self.slider_actual_pitch.poly.set_facecolor('red')
        elif pitch_error > 5:
            self.slider_actual_pitch.poly.set_facecolor('orange')
        else:
            self.slider_actual_pitch.poly.set_facecolor('blue')
            
    def _update_history(self):
        """Update the time history plots"""
        # Add current time
        if len(self.time_history) == 0:
            self.time_history.append(0)
        else:
            self.time_history.append(self.time_history[-1] + 1)
            
        # Add values to history
        self.target_roll_history.append(self.current_target_roll)
        self.target_pitch_history.append(self.current_target_pitch)
        self.actual_roll_history.append(self.current_actual_roll)
        self.actual_pitch_history.append(self.current_actual_pitch)
        self.throttle_history.append(self.current_target_throttle)
        
        # Limit history length
        if len(self.time_history) > self.history_length:
            self.time_history.pop(0)
            self.target_roll_history.pop(0)
            self.target_pitch_history.pop(0)
            self.actual_roll_history.pop(0)
            self.actual_pitch_history.pop(0)
            self.throttle_history.pop(0)
            
        # Update plot data
        self.line_target_roll.set_data(self.time_history, self.target_roll_history)
        self.line_actual_roll.set_data(self.time_history, self.actual_roll_history)
        self.line_target_pitch.set_data(self.time_history, self.target_pitch_history)
        self.line_actual_pitch.set_data(self.time_history, self.actual_pitch_history)
        
        # Adjust x-axis limits
        if len(self.time_history) > 1:
            self.ax_roll.set_xlim(self.time_history[0], self.time_history[-1])
            self.ax_pitch.set_xlim(self.time_history[0], self.time_history[-1])
            
    def clear_history(self):
        """Clear the time history"""
        self.time_history.clear()
        self.target_roll_history.clear()
        self.target_pitch_history.clear()
        self.actual_roll_history.clear()
        self.actual_pitch_history.clear()
        self.throttle_history.clear()
        
    def show(self):
        """Show the window"""
        plt.show()
        
    def close(self):
        """Close the window"""
        plt.close(self.fig)
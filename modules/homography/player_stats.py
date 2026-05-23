from collections import defaultdict, deque
import numpy as np
import logging

class PlayerStats:
    """Class to track player statistics during a match"""
    def __init__(self, player_id, team=None):
        self.player_id = player_id
        self.team = team
        self.jersey_number = None
        self.position = None  # Current position on field
        self.heatmap = defaultdict(int)  # Grid position -> count
        
        # Match statistics
        self.possession_time = 0
        self.distance_covered = 0
        self.passes_completed = 0
        self.passes_attempted = 0
        self.shots = 0
        self.shots_on_target = 0
        self.tackles = 0
        self.fouls = 0
        self.previous_positions = deque(maxlen=30)  # Store recent positions for velocity
        self.last_update_time = 0
        
    def update_position(self, x, y, timestamp):
        """Update player position and calculate stats"""
        # Store new position
        new_pos = (x, y)
        if self.position:
            # Calculate distance covered since last update
            distance = np.sqrt((new_pos[0] - self.position[0])**2 + (new_pos[1] - self.position[1])**2)
            time_delta = timestamp - self.last_update_time
            if time_delta > 0:
                self.distance_covered += distance
                
        self.position = new_pos
        self.previous_positions.append((new_pos, timestamp))
        self.last_update_time = timestamp
        
        # Update heatmap (discretize position to 10x10 grid)
        grid_x = int(x / 10)
        grid_y = int(y / 10)
        self.heatmap[(grid_x, grid_y)] += 1
    
    def calculate_velocity(self):
        """Calculate player's current velocity"""
        if len(self.previous_positions) < 2:
            return 0, 0
        
        # Get last two positions with timestamps
        (pos2, time2) = self.previous_positions[-1]
        (pos1, time1) = self.previous_positions[-2]
        
        # Calculate velocity components
        time_delta = time2 - time1
        if time_delta <= 0:
            return 0, 0
            
        velocity_x = (pos2[0] - pos1[0]) / time_delta
        velocity_y = (pos2[1] - pos1[1]) / time_delta
        
        return velocity_x, velocity_y
    
    def record_event(self, event_type):
        """Record statistical events"""
        if event_type == 'pass_attempt':
            self.passes_attempted += 1
        elif event_type == 'pass_complete':
            self.passes_completed += 1
        elif event_type == 'shot':
            self.shots += 1
        elif event_type == 'shot_on_target':
            self.shots_on_target += 1
        elif event_type == 'tackle':
            self.tackles += 1
        elif event_type == 'foul':
            self.fouls += 1
    
    def add_possession_time(self, seconds):
        """Add time to player's possession counter"""
        self.possession_time += seconds
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'player_id': self.player_id,
            'team': self.team,
            'jersey_number': self.jersey_number,
            'position': self.position,
            'possession_time': round(self.possession_time, 1),
            'distance_covered': round(self.distance_covered, 1),
            'passes': {
                'attempted': self.passes_attempted,
                'completed': self.passes_completed,
                'accuracy': round(self.passes_completed / max(1, self.passes_attempted) * 100, 1)
            },
            'shots': self.shots,
            'shots_on_target': self.shots_on_target,
            'tackles': self.tackles,
            'fouls': self.fouls
        } 
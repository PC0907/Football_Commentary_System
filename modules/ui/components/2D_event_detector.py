import numpy as np
import cv2
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional

class FootballEventDetector:
    """
    Rule-based football event detector operating on world-coordinate tracking data.

    All speed thresholds are specified in **m/s** and converted to m/frame
    internally using *fps*.  This prevents the previous bug where thresholds
    were implicitly in m/frame, making them 30× too tight at 30 FPS
    (e.g. ``ball_speed > 5`` m/frame ≡ 150 m/s — physically impossible).

    Threshold rationale (FIFA-typical values, tuneable):
    ─────────────────────────────────────────────────────────────────────────
    THRESHOLD                       VALUE   RATIONALE
    shot_speed_ms                   8 m/s   Minimum credible shot speed (~29 km/h)
    free_kick_speed_ms              3 m/s   Ball resumes movement after set piece
    stationary_speed_ms             0.3 m/s Ball is considered stationary below this
    min_pass_distance               3 m     Ignore micro-movements within possession
    shot_distance_threshold         20 m    Must be within 20 m of goal to be a shot
    possession_radius_m             1.5 m   Ball "owned" if nearest player is <1.5 m
    foul_proximity_m                2.0 m   Players must be <2 m apart for foul check
    foul_ball_proximity_m           3.0 m   Ball must be <3 m from fouled player
    ─────────────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        field_dimensions=(105, 68),
        fps: float = 30.0,
        min_pass_distance: float = 3.0,
        shot_distance_threshold: float = 20.0,
        shot_speed_ms: float = 8.0,
        free_kick_speed_ms: float = 3.0,
        stationary_speed_ms: float = 0.3,
        possession_radius_m: float = 1.5,
        foul_proximity_m: float = 2.0,
        foul_ball_proximity_m: float = 3.0,
    ):
        """
        Parameters
        ----------
        field_dimensions        : (length, width) in metres
        fps                     : video frame rate — used to convert m/s → m/frame
        min_pass_distance       : min ball travel (m) to register as a pass
        shot_distance_threshold : max distance from goal (m) to register as a shot
        shot_speed_ms           : min ball speed (m/s) to register as a shot
        free_kick_speed_ms      : min ball speed (m/s) to register a free-kick restart
        stationary_speed_ms     : max ball speed (m/s) below which ball is "stationary"
        possession_radius_m     : radius (m) within which a player "has" the ball
        foul_proximity_m        : max inter-player distance (m) for foul detection
        foul_ball_proximity_m   : max ball-to-fouled-player distance (m) to confirm foul
        """
        self.field_length, self.field_width = field_dimensions
        self.fps = max(fps, 1.0)
        self.min_pass_distance       = min_pass_distance
        self.shot_distance_threshold = shot_distance_threshold
        self.possession_radius_m     = possession_radius_m
        self.foul_proximity_m        = foul_proximity_m
        self.foul_ball_proximity_m   = foul_ball_proximity_m

        # Convert m/s thresholds to m/frame for per-frame speed comparisons
        self.shot_speed_mf        = shot_speed_ms        / self.fps
        self.free_kick_speed_mf   = free_kick_speed_ms   / self.fps
        self.stationary_speed_mf  = stationary_speed_ms  / self.fps

        # Define field zones and landmarks
        self.zones = self._define_field_zones()

        # Goal centre positions
        self.goals = {
            "home": (0,                 self.field_width / 2),
            "away": (self.field_length, self.field_width / 2),
        }

        # Event history
        self.events = []

        # Detection state
        self.ball_possession       = None   # player_id with the ball
        self.ball_possession_team  = None   # team_id with the ball
        self.last_ball_contact     = None
        self.ball_stationary_frames = 0
        self.stationary_threshold  = 5      # consecutive frames to call ball stationary
        self.ball_speed_samples: list = []
        self.ball_speed_window     = 10
        self.is_play_active        = True

        # Previous-frame state
        self.prev_ball_pos    = None
        self.prev_players_pos = {}

    def _define_field_zones(self) -> Dict[str, Dict]:
        """Define key zones on the football field"""
        half_length = self.field_length / 2
        half_width = self.field_width / 2
        
        # Define penalty areas (16m from goal line, 40.3m wide)
        penalty_area_width = 40.3
        penalty_area_length = 16.5
        
        # Define goal areas (5.5m from goal line, 18.3m wide)
        goal_area_width = 18.3
        goal_area_length = 5.5
        
        zones = {
            # Home half
            'home_half': {
                'x_min': 0, 
                'x_max': half_length, 
                'y_min': 0, 
                'y_max': self.field_width
            },
            
            # Away half
            'away_half': {
                'x_min': half_length, 
                'x_max': self.field_length, 
                'y_min': 0, 
                'y_max': self.field_width
            },
            
            # Home penalty area
            'home_penalty_area': {
                'x_min': 0, 
                'x_max': penalty_area_length, 
                'y_min': half_width - penalty_area_width/2, 
                'y_max': half_width + penalty_area_width/2
            },
            
            # Away penalty area
            'away_penalty_area': {
                'x_min': self.field_length - penalty_area_length, 
                'x_max': self.field_length, 
                'y_min': half_width - penalty_area_width/2, 
                'y_max': half_width + penalty_area_width/2
            },
            
            # Home goal area
            'home_goal_area': {
                'x_min': 0, 
                'x_max': goal_area_length, 
                'y_min': half_width - goal_area_width/2, 
                'y_max': half_width + goal_area_width/2
            },
            
            # Away goal area
            'away_goal_area': {
                'x_min': self.field_length - goal_area_length, 
                'x_max': self.field_length, 
                'y_min': half_width - goal_area_width/2, 
                'y_max': half_width + goal_area_width/2
            },
            
            # Corner areas (5m radius from corner)
            'home_left_corner': {
                'center': (0, 0),
                'radius': 5
            },
            'home_right_corner': {
                'center': (0, self.field_width),
                'radius': 5
            },
            'away_left_corner': {
                'center': (self.field_length, 0),
                'radius': 5
            },
            'away_right_corner': {
                'center': (self.field_length, self.field_width),
                'radius': 5
            }
        }
        
        return zones

    def _is_in_zone(self, position: Tuple[float, float], zone: Dict) -> bool:
        """Check if a position is within a defined zone"""
        x, y = position
        
        if 'radius' in zone:  # Circular zone (like corner areas)
            center_x, center_y = zone['center']
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            return distance <= zone['radius']
        else:  # Rectangular zone
            return (zone['x_min'] <= x <= zone['x_max'] and 
                    zone['y_min'] <= y <= zone['y_max'])

    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _calculate_ball_speed(self, pos1: Tuple[float, float], pos2: Tuple[float, float], frames_elapsed: int = 1) -> float:
        """Calculate ball speed in meters per frame"""
        if frames_elapsed == 0:
            return 0
        return self._calculate_distance(pos1, pos2) / frames_elapsed

    def _find_closest_player(self, players_pos: Dict, position: Tuple[float, float]) -> Tuple[str, str, float]:
        """Find the closest player to a given position
        
        Returns:
            Tuple of (player_id, team_id, distance)
        """
        min_distance = float('inf')
        closest_player = None
        closest_team = None
        
        for team_id, team_players in players_pos.items():
            for player_id, player_pos in team_players.items():
                distance = self._calculate_distance(player_pos, position)
                if distance < min_distance:
                    min_distance = distance
                    closest_player = player_id
                    closest_team = team_id
        
        return closest_player, closest_team, min_distance

    def _is_ball_stationary(self, current_ball_pos: Tuple[float, float]) -> bool:
        """
        Return True if the ball moved less than stationary_speed_mf metres since
        the previous frame.

        Threshold is fps-normalised at construction time, so this check is
        independent of video frame rate.
        """
        if self.prev_ball_pos is None:
            return False
        distance = self._calculate_distance(self.prev_ball_pos, current_ball_pos)
        return distance < self.stationary_speed_mf

    def _detect_ball_possession(self, ball_pos: Tuple[float, float], players_pos: Dict) -> Tuple[Optional[str], Optional[str]]:
        """
        Return (player_id, team_id) of the player in possession, or (None, None).

        A player is deemed in possession when they are within *possession_radius_m*
        of the ball (default 1.5 m).
        """
        closest_player, closest_team, distance = self._find_closest_player(players_pos, ball_pos)
        if distance <= self.possession_radius_m:
            return closest_player, closest_team
        return None, None

    def detect_pass(self, 
                   current_possession: Tuple[str, str], 
                   prev_possession: Tuple[str, str],
                   ball_pos: Tuple[float, float]) -> Optional[Dict]:
        """Detect if a pass has occurred"""
        if None in prev_possession or None in current_possession:
            return None
            
        prev_player, prev_team = prev_possession
        current_player, current_team = current_possession
        
        # If ball has moved from one player to another on the same team
        if (prev_player != current_player and prev_team == current_team and 
            self.prev_ball_pos is not None):
            
            distance = self._calculate_distance(self.prev_ball_pos, ball_pos)
            
            if distance >= self.min_pass_distance:
                return {
                    'type': 'pass',
                    'from_player': prev_player,
                    'to_player': current_player,
                    'team': prev_team,
                    'position': ball_pos,
                    'distance': distance
                }
        
        return None

    def detect_shot(self,
                   ball_pos: Tuple[float, float],
                   ball_speed: float,
                   current_possession: Tuple[str, str]) -> Optional[Dict]:
        """
        Detect a shot on goal.

        Conditions (all must hold):
        1. Ball is within *shot_distance_threshold* metres of the opponent goal.
        2. Ball speed exceeds *shot_speed_mf* metres/frame
           (≡ *shot_speed_ms* m/s at construction fps — default 8 m/s / ~29 km/h).
        3. Ball is in the attacking half for the team in possession.

        The previous threshold of ``ball_speed > 5`` m/frame was equivalent to
        150 m/s at 30 FPS — impossible — so no shots were ever detected.
        """
        if None in current_possession:
            return None

        player_id, team_id = current_possession
        goal_pos = self.goals["away"] if team_id == "home" else self.goals["home"]
        distance_to_goal = self._calculate_distance(ball_pos, goal_pos)

        in_attacking_half = (
            (team_id == "home" and ball_pos[0] > self.field_length / 2) or
            (team_id == "away" and ball_pos[0] < self.field_length / 2)
        )

        if (distance_to_goal <= self.shot_distance_threshold
                and ball_speed >= self.shot_speed_mf
                and in_attacking_half):
            return {
                "type":             "shot",
                "player":           player_id,
                "team":             team_id,
                "position":         ball_pos,
                "distance_to_goal": distance_to_goal,
                "ball_speed_ms":    ball_speed * self.fps,   # store in m/s for readability
            }
        return None

    def detect_goal(self, 
                   ball_pos: Tuple[float, float], 
                   shot_event: Dict) -> Optional[Dict]:
        """Detect if a goal has been scored from a shot"""
        if not shot_event:
            return None
        
        # Goal line positions
        home_goal_line = (0, self.field_width/2)
        away_goal_line = (self.field_length, self.field_width/2)
        
        player_id = shot_event['player']
        team_id = shot_event['team']
        
        # Check if ball is very close to goal line
        if team_id == 'home':
            distance_to_goal_line = self._calculate_distance(ball_pos, away_goal_line)
            defending_team = 'away'
        else:
            distance_to_goal_line = self._calculate_distance(ball_pos, home_goal_line)
            defending_team = 'home'
        
        if distance_to_goal_line < 1.5:  # Ball is very close to goal line
            # Check if ball is within goal width (7.32m)
            goal_y_center = self.field_width / 2
            if abs(ball_pos[1] - goal_y_center) <= 3.66:  # Half of goal width
                return {
                    'type': 'goal',
                    'player': player_id,
                    'team': team_id,
                    'position': ball_pos,
                    'against': defending_team
                }
        
        return None

    def detect_corner_kick(self, 
                          ball_pos: Tuple[float, float],
                          is_ball_stationary: bool) -> Optional[Dict]:
        """Detect if a corner kick is being taken"""
        if not is_ball_stationary:
            return None
        
        # Check if ball is in one of the corner areas
        corner_zones = ['home_left_corner', 'home_right_corner', 
                        'away_left_corner', 'away_right_corner']
        
        for corner_name in corner_zones:
            if self._is_in_zone(ball_pos, self.zones[corner_name]):
                # Determine attacking team (opposite of the half)
                if 'home' in corner_name:
                    attacking_team = 'away'
                else:
                    attacking_team = 'home'
                
                return {
                    'type': 'corner_kick',
                    'team': attacking_team,
                    'position': ball_pos,
                    'corner': corner_name
                }
        
        return None

    def detect_free_kick(self,
                        ball_pos: Tuple[float, float],
                        is_ball_stationary: bool,
                        ball_speed: float) -> Optional[Dict]:
        """
        Detect a free-kick restart: ball was stationary for ≥ stationary_threshold
        frames and then suddenly accelerates past free_kick_speed_mf metres/frame
        (≡ free_kick_speed_ms m/s — default 3 m/s).

        The previous threshold of ``ball_speed > 3`` m/frame ≡ 90 m/s — never fired.
        """
        if self.ball_stationary_frames < self.stationary_threshold:
            return None

        if ball_speed >= self.free_kick_speed_mf:
            closest_player, closest_team, _ = self._find_closest_player(
                self.prev_players_pos, self.prev_ball_pos
            )
            if closest_player and closest_team:
                return {
                    "type":     "free_kick",
                    "team":     closest_team,
                    "player":   closest_player,
                    "position": ball_pos,
                }
        return None

    def detect_foul(self, 
                   players_pos: Dict,
                   ball_pos: Tuple[float, float],
                   is_play_active: bool,
                   player_velocities: Dict) -> Optional[Dict]:
        """Detect potential fouls based on player collisions and sudden stops
        
        This is a simplified detection that looks for:
        1. Proximity between players from different teams
        2. Sudden change in player velocity
        3. Ball stationary after the collision
        """
        if not is_play_active or self.prev_players_pos == {}:
            return None
        
        potential_fouls = []

        # Check for player collisions
        # foul_proximity_m default = 2.0 m  (was 1.0 m — too tight for noisy homography)
        for team1, team1_players in players_pos.items():
            for team2, team2_players in players_pos.items():
                if team1 == team2:
                    continue  # Skip same-team contacts

                for player1_id, player1_pos in team1_players.items():
                    for player2_id, player2_pos in team2_players.items():
                        distance = self._calculate_distance(player1_pos, player2_pos)

                        if distance < self.foul_proximity_m:
                            # Check if one player had sudden velocity change
                            if (player1_id in player_velocities and 
                                player_velocities[player1_id]['magnitude'] > 2.0 and
                                player_velocities[player1_id]['change'] < -1.5):
                                
                                potential_fouls.append({
                                    'fouling_player': player2_id,
                                    'fouled_player': player1_id,
                                    'fouling_team': team2,
                                    'fouled_team': team1,
                                    'severity': player_velocities[player1_id]['change'] * -1
                                })
                                
                            elif (player2_id in player_velocities and 
                                  player_velocities[player2_id]['magnitude'] > 2.0 and
                                  player_velocities[player2_id]['change'] < -1.5):
                                
                                potential_fouls.append({
                                    'fouling_player': player1_id,
                                    'fouled_player': player2_id,
                                    'fouling_team': team1,
                                    'fouled_team': team2,
                                    'severity': player_velocities[player2_id]['change'] * -1
                                })
        
        if not potential_fouls:
            return None
        
        # Choose the most severe potential foul
        most_severe = max(potential_fouls, key=lambda x: x['severity'])
        
        # Check if ball is near the foul
        fouled_player_pos = players_pos[most_severe['fouled_team']][most_severe['fouled_player']]
        distance_to_ball = self._calculate_distance(fouled_player_pos, ball_pos)
        
        if distance_to_ball < self.foul_ball_proximity_m:
            foul_event = {
                'type': 'foul',
                'fouling_player': most_severe['fouling_player'],
                'fouled_player': most_severe['fouled_player'],
                'fouling_team': most_severe['fouling_team'],
                'fouled_team': most_severe['fouled_team'],
                'position': fouled_player_pos,
                'severity': most_severe['severity']
            }
            
            # Determine if this might be a penalty
            if self._is_in_zone(fouled_player_pos, self.zones['home_penalty_area']) and most_severe['fouled_team'] == 'away':
                foul_event['penalty'] = True
                foul_event['type'] = 'penalty'
            elif self._is_in_zone(fouled_player_pos, self.zones['away_penalty_area']) and most_severe['fouled_team'] == 'home':
                foul_event['penalty'] = True
                foul_event['type'] = 'penalty'
            else:
                foul_event['penalty'] = False
            
            return foul_event
        
        return None

    def _calculate_player_velocities(self, current_players_pos: Dict) -> Dict:
        """Calculate player velocities and velocity changes"""
        velocities = {}
        
        if not self.prev_players_pos:
            return velocities
        
        for team_id, team_players in current_players_pos.items():
            if team_id not in self.prev_players_pos:
                continue
                
            for player_id, current_pos in team_players.items():
                if player_id not in self.prev_players_pos[team_id]:
                    continue
                    
                prev_pos = self.prev_players_pos[team_id][player_id]
                current_velocity = self._calculate_distance(prev_pos, current_pos)
                
                velocities[player_id] = {
                    'magnitude': current_velocity,
                    'change': 0  # Default value
                }
                
                # If we have previous velocity data, calculate change
                if hasattr(self, 'player_velocities') and player_id in self.player_velocities:
                    prev_velocity = self.player_velocities[player_id]['magnitude']
                    velocities[player_id]['change'] = current_velocity - prev_velocity
        
        return velocities

    def process_frame(self, 
                     frame_number: int,
                     ball_pos: Tuple[float, float],
                     players_pos: Dict[str, Dict[str, Tuple[float, float]]]) -> List[Dict]:
        """
        Process a single frame to detect events
        
        Args:
            frame_number: Current frame number
            ball_pos: Position of the ball (x, y) in meters
            players_pos: Dictionary mapping team_id -> {player_id -> (x, y)}
        
        Returns:
            List of detected events
        """
        frame_events = []
        
        # Calculate player velocities
        player_velocities = self._calculate_player_velocities(players_pos)
        
        # Check if ball is stationary
        is_ball_stationary = self._is_ball_stationary(ball_pos)
        if is_ball_stationary:
            self.ball_stationary_frames += 1
        else:
            self.ball_stationary_frames = 0
        
        # Calculate ball speed
        ball_speed = 0
        if self.prev_ball_pos is not None:
            ball_speed = self._calculate_ball_speed(self.prev_ball_pos, ball_pos)
            
            # Store ball speed for spike detection
            self.ball_speed_samples.append(ball_speed)
            if len(self.ball_speed_samples) > self.ball_speed_window:
                self.ball_speed_samples.pop(0)
        
        # Detect ball possession
        current_player, current_team = self._detect_ball_possession(ball_pos, players_pos)
        current_possession = (current_player, current_team)
        prev_possession = (self.ball_possession, self.ball_possession_team)
        
        # Detect events
        
        # 1. Pass detection
        pass_event = self.detect_pass(current_possession, prev_possession, ball_pos)
        if pass_event:
            pass_event['frame'] = frame_number
            frame_events.append(pass_event)
        
        # 2. Shot detection
        shot_event = self.detect_shot(ball_pos, ball_speed, prev_possession)
        if shot_event:
            shot_event['frame'] = frame_number
            frame_events.append(shot_event)
            
            # 3. Goal detection (only check if there was a shot)
            goal_event = self.detect_goal(ball_pos, shot_event)
            if goal_event:
                goal_event['frame'] = frame_number
                frame_events.append(goal_event)
        
        # 4. Corner kick detection
        corner_event = self.detect_corner_kick(ball_pos, is_ball_stationary)
        if corner_event:
            corner_event['frame'] = frame_number
            frame_events.append(corner_event)
        
        # 5. Free kick detection
        free_kick_event = self.detect_free_kick(ball_pos, is_ball_stationary, ball_speed)
        if free_kick_event:
            free_kick_event['frame'] = frame_number
            frame_events.append(free_kick_event)
        
        # 6. Foul detection
        foul_event = self.detect_foul(players_pos, ball_pos, self.is_play_active, player_velocities)
        if foul_event:
            foul_event['frame'] = frame_number
            frame_events.append(foul_event)
            self.is_play_active = False  # Play stops after a foul
        
        # Update state for next frame
        self.prev_ball_pos = ball_pos
        self.prev_players_pos = players_pos
        self.ball_possession = current_player
        self.ball_possession_team = current_team
        self.player_velocities = player_velocities
        
        # Update global events list
        self.events.extend(frame_events)
        
        return frame_events

    def get_all_events(self) -> List[Dict]:
        """Get all detected events"""
        return self.events

    def reset(self):
        """Reset the detector state"""
        self.events = []
        self.ball_possession = None
        self.ball_possession_team = None
        self.prev_ball_pos = None
        self.prev_players_pos = {}
        self.ball_stationary_frames = 0
        self.ball_speed_samples = []
        self.is_play_active = True


class PlayerStatsTracker:
    def __init__(self, team_sheet: Dict[str, Dict[str, Dict]]):
        """
        Initialize the player stats tracker
        
        Args:
            team_sheet: Dictionary with team info and player details
                {
                    'team1': {
                        'player1': {
                            'name': 'Player Name',
                            'number': 10,
                            'position': 'MF'
                        },
                        ...
                    },
                    'team2': {
                        ...
                    }
                }
        """
        self.team_sheet = team_sheet
        self.stats = self._initialize_stats()
        
        # Track heatmap data
        self.position_samples = {team_id: {player_id: [] for player_id in team_players}
                                for team_id, team_players in team_sheet.items()}

    def _initialize_stats(self) -> Dict:
        """Initialize stats dictionary for all teams and players"""
        stats = {}
        
        # Team-level stats
        for team_id in self.team_sheet.keys():
            stats[team_id] = {
                'goals': 0,
                'shots': 0,
                'shots_on_target': 0,
                'passes': 0,
                'pass_accuracy': 0.0,
                'possession': 0.0,
                'corners': 0,
                'free_kicks': 0,
                'fouls_committed': 0,
                'fouls_received': 0,
                'possession_frames': 0,
                'successful_passes': 0
            }
            
            # Player-level stats
            stats[team_id]['players'] = {}
            
            for player_id, player_info in self.team_sheet[team_id].items():
                stats[team_id]['players'][player_id] = {
                    'name': player_info['name'],
                    'number': player_info['number'],
                    'position': player_info['position'],
                    'goals': 0,
                    'assists': 0,
                    'shots': 0,
                    'shots_on_target': 0,
                    'passes': 0,
                    'successful_passes': 0,
                    'pass_accuracy': 0.0,
                    'distance_covered': 0.0,
                    'fouls_committed': 0,
                    'fouls_received': 0,
                    'possession_frames': 0,
                    'heatmap': []
                }
        
        return stats

    def update_positions(self, frame_number: int, players_pos: Dict[str, Dict[str, Tuple[float, float]]]):
        """Update player position data for heatmaps and distance tracking"""
        for team_id, team_players in players_pos.items():
            for player_id, position in team_players.items():
                if team_id in self.position_samples and player_id in self.position_samples[team_id]:
                    # Add position sample for heatmap
                    self.position_samples[team_id][player_id].append({
                        'frame': frame_number,
                        'position': position
                    })
                    
                    # Calculate distance traveled since last frame
                    samples = self.position_samples[team_id][player_id]
                    if len(samples) >= 2:
                        prev_pos = samples[-2]['position']
                        curr_pos = position
                        
                        distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                                           (curr_pos[1] - prev_pos[1])**2)
                        
                        # Update distance covered
                        if team_id in self.stats and player_id in self.stats[team_id]['players']:
                            self.stats[team_id]['players'][player_id]['distance_covered'] += distance

    def process_events(self, events: List[Dict], ball_possession: Tuple[str, str]):
        """
        Update player and team statistics based on events and possession
        
        Args:
            events: List of events detected in the current frame
            ball_possession: Tuple of (player_id, team_id) who has possession
        """
        # Update possession stats
        player_id, team_id = ball_possession
        if player_id is not None and team_id is not None:
            if team_id in self.stats:
                self.stats[team_id]['possession_frames'] += 1
                
                if player_id in self.stats[team_id]['players']:
                    self.stats[team_id]['players'][player_id]['possession_frames'] += 1
        
        # Process each event
        for event in events:
            event_type = event['type']
            
            if event_type == 'pass':
                team_id = event['team']
                from_player = event['from_player']
                to_player = event['to_player']
                
                # Update team stats
                if team_id in self.stats:
                    self.stats[team_id]['passes'] += 1
                    self.stats[team_id]['successful_passes'] += 1
                
                # Update player stats
                if team_id in self.stats and from_player in self.stats[team_id]['players']:
                    self.stats[team_id]['players'][from_player]['passes'] += 1
                    self.stats[team_id]['players'][from_player]['successful_passes'] += 1
                    
                    # Calculate updated pass accuracy
                    passes = self.stats[team_id]['players'][from_player]['passes']
                    successful = self.stats[team_id]['players'][from_player]['successful_passes']
                    if passes > 0:
                        accuracy = (successful / passes) * 100
                        self.stats[team_id]['players'][from_player]['pass_accuracy'] = accuracy
            
            elif event_type == 'shot':
                team_id = event['team']
                player_id = event['player']
                
                # Update team stats
                if team_id in self.stats:
                    self.stats[team_id]['shots'] += 1
                
                # Update player stats
                if team_id in self.stats and player_id in self.stats[team_id]['players']:
                    self.stats[team_id]['players'][player_id]['shots'] += 1
            
            elif event_type == 'goal':
                team_id = event['team']
                player_id = event['player']
                
                # Update team stats
                if team_id in self.stats:
                    self.stats[team_id]['goals'] += 1
                    self.stats[team_id]['shots'] += 1
                    self.stats[team_id]['shots_on_target'] += 1
                
                # Update player stats
                if team_id in self.stats and player_id in self.stats[team_id]['players']:
                    self.stats[team_id]['players'][player_id]['goals'] += 1
                    self.stats[team_id]['players'][player_id]['shots'] += 1
                    self.stats[team_id]['players'][player_id]['shots_on_target'] += 1
            
            elif event_type == 'corner_kick':
                team_id = event['team']
                
                # Update team stats
                if team_id in self.stats:
                    self.stats[team_id]['corners'] += 1
            
            elif event_type == 'free_kick':
                team_id = event['team']
                
                # Update team stats
                if team_id in self.stats:
                    self.stats[team_id]['free_kicks'] += 1
            
            elif event_type == 'foul' or event_type == 'penalty':
                fouling_team = event['fouling_team']
                fouled_team = event['fouled_team']
                fouling_player = event['fouling_player']
                fouled_player = event['fouled_player']
                
                # Update team stats
                if fouling_team in self.stats:
                    self.stats[fouling_team]['fouls_committed'] += 1
                if fouled_team in self.stats:
                    self.stats[fouled_team]['fouls_received'] += 1
                
                # Update player stats
                if (fouling_team in self.stats and 
                    fouling_player in self.stats[fouling_team]['players']):
                    self.stats[fouling_team]['players'][fouling_player]['fouls_committed'] += 1
                
                if (fouled_team in self.stats and 
                    fouled_player in self.stats[fouled_team]['players']):
                    self.stats[fouled_team]['players'][fouled_player]['fouls_received'] += 1

    def calculate_possession_percentages(self, total_frames: int) -> Dict[str, float]:
        """Calculate possession percentages for each team"""
        if total_frames == 0:
            return {}
            
        possession = {}
        for team_id, team_stats in self.stats.items():
            possession[team_id] = (team_stats['possession_frames'] / total_frames) * 100
        
        return possession

    def calculate_team_pass_accuracy(self) -> Dict[str, float]:
        """Calculate pass accuracy for each team"""
        accuracy = {}
        for team_id, team_stats in self.stats.items():
            passes = team_stats['passes']
            successful = team_stats['successful_passes']
            if passes > 0:
                accuracy[team_id] = (successful / passes) * 100
            else:
                accuracy[team_id] = 0.0
        
        return accuracy

    def generate_heatmaps(self) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
        """Generate heatmap data for all players"""
        heatmaps = {}
        for team_id, team_players in self.position_samples.items():
            heatmaps[team_id] = {}
            for player_id, samples in team_players.items():
                positions = [sample['position'] for sample in samples]
                heatmaps[team_id][player_id] = positions
                
                # Also store in stats
                if (team_id in self.stats and 
                    player_id in self.stats[team_id]['players']):
                    self.stats[team_id]['players'][player_id]['heatmap'] = positions
        
        return heatmaps

    def get_player_stats(self, team_id: str, player_id: str) -> Dict[str, Any]:
        """Get statistics for a specific player"""
        if (team_id in self.stats and 
            player_id in self.stats[team_id]['players']):
            return self.stats[team_id]['players'][player_id]
        return {}

    def get_team_stats(self, team_id: str) -> Dict[str, Any]:
        """Get statistics for a specific team"""
        if team_id in self.stats:
            return self.stats[team_id]
        return {}

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get all statistics for all teams and players"""
        return self.stats

    def reset(self):
        """Reset all statistics"""
        self.stats = self._initialize_stats()
        self.position_samples = {team_id: {player_id: [] for player_id in team_players}
                                 for team_id, team_players in self.team_sheet.items()}


class FootballAnalysisPipeline:
    def __init__(self, team_sheet: Dict[str, Dict[str, Dict]], field_dimensions=(105, 68)):
        """
        Initialize the analysis pipeline
        
        Args:
            team_sheet: Dictionary with team and player information
            field_dimensions: Tuple of (length, width) in meters
        """
        self.event_detector = FootballEventDetector(field_dimensions=field_dimensions)
        self.stats_tracker = PlayerStatsTracker(team_sheet)
        self.frame_count = 0

    def process_frame(self, 
                     ball_pos: Tuple[float, float], 
                     players_pos: Dict[str, Dict[str, Tuple[float, float]]]) -> List[Dict]:
        """
        Process a single frame of tracking data
        
        Args:
            ball_pos: Position of the ball (x, y) in meters
            players_pos: Dictionary mapping team_id -> {player_id -> (x, y)}
        
        Returns:
            List of detected events
        """
        self.frame_count += 1
        
        # Detect events
        events = self.event_detector.process_frame(self.frame_count, ball_pos, players_pos)
        
        # Update player positions for heatmaps and distance
        self.stats_tracker.update_positions(self.frame_count, players_pos)
        
        # Update statistics based on events and possession
        possession = (self.event_detector.ball_possession, 
                      self.event_detector.ball_possession_team)
        self.stats_tracker.process_events(events, possession)
        
        return events

    def get_match_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive match summary"""
        summary = {
            'frame_count': self.frame_count,
            'events': self.event_detector.get_all_events(),
            'stats': self.stats_tracker.get_all_stats(),
            'possession': self.stats_tracker.calculate_possession_percentages(self.frame_count),
            'pass_accuracy': self.stats_tracker.calculate_team_pass_accuracy(),
            'heatmaps': self.stats_tracker.generate_heatmaps()
        }
        return summary

    def reset(self):
        """Reset the entire pipeline"""
        self.event_detector.reset()
        self.stats_tracker.reset()
        self.frame_count = 0
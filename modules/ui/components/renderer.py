import cv2
import numpy as np
from typing import Dict, Any

class Renderer:
    """Handles rendering of processed frames with overlays"""
    
    def __init__(self):
        self.colors = {
            'team_a': (0, 0, 255),  # Blue
            'team_b': (255, 0, 0),  # Red
            'ball': (0, 255, 0),    # Green
            'text': (255, 255, 255) # White
        }
        
    def render(self, frame: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Render the frame with all overlays
        
        Args:
            frame: Input frame
            metadata: Dictionary containing frame metadata and processing results
            
        Returns:
            Rendered frame with overlays
        """
        # Create a copy of the frame
        rendered = frame.copy()
        
        # Draw player bounding boxes and IDs
        if 'tracking' in metadata:
            for player_id, bbox in metadata['tracking']['players'].items():
                x1, y1, x2, y2 = bbox
                team = metadata.get('teams', {}).get(player_id, 'unknown')
                color = self.colors.get(team, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
                
                # Draw player ID
                cv2.putText(
                    rendered,
                    f"ID:{player_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
                
                # Draw jersey number if available
                if 'jersey_numbers' in metadata and player_id in metadata['jersey_numbers']:
                    jersey = metadata['jersey_numbers'][player_id]
                    cv2.putText(
                        rendered,
                        f"#{jersey}",
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )
        
        # Draw ball if detected
        if 'tracking' in metadata and 'ball' in metadata['tracking']:
            ball_x, ball_y, ball_w, ball_h = metadata['tracking']['ball']
            cv2.circle(
                rendered,
                (int(ball_x + ball_w/2), int(ball_y + ball_h/2)),
                5,
                self.colors['ball'],
                -1
            )
        
        # Draw field positions
        if 'field_positions' in metadata:
            for pos in metadata['field_positions']:
                x, y = int(pos['world_x_meters']), int(pos['world_y_meters'])
                cv2.circle(rendered, (x, y), 3, (0, 255, 0), -1)
        
        # Draw match time
        if 'timestamp' in metadata:
            minutes = int(metadata['timestamp'] // 60)
            seconds = int(metadata['timestamp'] % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            cv2.putText(
                rendered,
                time_str,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                self.colors['text'],
                2
            )
        
        # Draw score if available
        if 'score' in metadata:
            score = metadata['score']
            score_str = f"{score['team_a']} - {score['team_b']}"
            cv2.putText(
                rendered,
                score_str,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                self.colors['text'],
                2
            )
        
        # Draw commentary if available
        if 'commentary' in metadata:
            commentary = metadata['commentary']
            cv2.putText(
                rendered,
                commentary,
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.colors['text'],
                2
            )
        
        return rendered 
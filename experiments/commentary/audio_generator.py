import json
import random
import csv
import os
import re
from datetime import datetime, timedelta
import subprocess
import pyttsx3
from pydub import AudioSegment  # Required import for audio handling
import os

class CommentaryGenerator:
    def __init__(self, json_file_path, player_csv_path=None,team_names=None, confidence_threshold=0.8):
        """
        Initialize the commentary generator.
        
        Args:
            json_file_path: Path to the JSON file with event data
            player_csv_path: Path to the CSV file with player jersey numbers (optional)
            team_names: Dictionary mapping 'left' and 'right' to team names
            confidence_threshold: Minimum confidence score to consider an event valid
        """
        self.json_file_path = json_file_path
        self.player_csv_path = player_csv_path
        self.team_names = team_names or {"left": "Home Team", "right": "Away Team"}
        self.confidence_threshold = confidence_threshold
        
        # Templates for each event type
        self.templates = {
            "PASS": [
                "{player} passes the ball to {secondary_player}.",
                "Nice pass from {player} to {secondary_player}.",
                "{player} finds {secondary_player} with a good pass.",
                "The ball is played by {player} to {secondary_player}.",
                "{player} with a clever pass to {secondary_player}."
            ],
            "HIGH PASS": [
                "{player} sends a high ball to {secondary_player}.",
                "A lofted pass from {player} looking for {secondary_player}.",
                "{player} spots {secondary_player} and delivers a high pass.",
                "Great vision from {player} with that high pass to {secondary_player}.",
                "{player} launches the ball toward {secondary_player}."
            ],
            "DRIVE": [
                "{player} drives forward with the ball.",
                "{player} makes a good run with possession.",
                "Look at {player} driving into space.",
                "Powerful drive from {player}.",
                "{player} showing good pace on the ball."
            ],
            "HEADER": [
                "{player} connects with a header.",
                "A headed effort from {player}.",
                "{player} rises high to head the ball.",
                "Good header from {player}.",
                "{player} gets up well for that header."
            ],
            "CROSS": [
                "{player} delivers a cross into the box.",
                "A cross comes in from {player}.",
                "{player} whips the ball into the danger area.",
                "Good cross from {player}.",
                "{player} with a searching ball into the box."
            ],
            "SHOT": [
                "{player} takes a shot!",
                "A shot from {player}!",
                "{player} tries his luck!",
                "An effort on goal from {player}!",
                "{player} pulls the trigger!"
            ],
            "BALL PLAYER BLOCK": [
                "{player} blocks the ball.",
                "Good block from {player}.",
                "{player} gets in the way of that one.",
                "Blocked by {player}.",
                "{player} with a crucial block."
            ],
            "OUT": [
                "The ball goes out of play.",
                "That's out of bounds.",
                "The ball crosses the line.",
                "And that's gone out.",
                "Play will restart with the ball having gone out."
            ],
            "THROW IN": [
                "Throw-in for {team_name}.",
                "{team_name} to take the throw-in.",
                "The ball will be thrown back in by {team_name}.",
                "We'll have a throw-in for {team_name}.",
                "{team_name} will restart play with a throw."
            ],
            "PLAYER SUCCESSFUL TACKLE": [
                "Great tackle by {player}!",
                "{player} wins back possession with a clean tackle.",
                "Well-timed challenge from {player}.",
                "{player} with an excellent tackle.",
                "Superb defensive work by {player}."
            ],
            "Corner": [
                "Corner kick for {team_name}.",
                "{team_name} have a corner opportunity.",
                "The referee signals for a corner to {team_name}.",
                "It's a corner for {team_name}.",
                "Corner kick coming up for {team_name}."
            ]
        }
        
        # Generic templates that don't require player names
        self.generic_templates = {
            "PASS": [
                "A pass is made to a teammate.",
                "The ball is moved forward.",
                "A good passing move.",
                "The team maintains possession with a pass.",
                "A simple pass to keep the ball moving."
            ],
            "HIGH PASS": [
                "A high ball is played forward.",
                "The ball is lofted into space.",
                "A high pass to change the point of attack.",
                "A lofted ball over the defense.",
                "The ball is sent high across the field."
            ],
            "DRIVE": [
                "A driving run with the ball.",
                "Good forward movement with possession.",
                "A direct run into space.",
                "Pushing forward with the ball.",
                "A determined run with the ball."
            ],
            "HEADER": [
                "The ball is headed clear.",
                "A headed attempt.",
                "Rising high for the header.",
                "A good header under pressure.",
                "Meeting the ball with a header."
            ],
            "CROSS": [
                "A cross into the danger area.",
                "The ball is whipped into the box.",
                "A delivery into the penalty area.",
                "A cross looking for attackers in the box.",
                "The ball is sent into the mixer."
            ],
            "SHOT": [
                "A shot on goal!",
                "An effort toward the target!",
                "Trying for goal!",
                "A strike at goal!",
                "Going for glory with that shot!"
            ],
            "BALL PLAYER BLOCK": [
                "The shot is blocked.",
                "A crucial block to deny the attempt.",
                "The defense stands firm with a block.",
                "Blocked by the defender.",
                "The ball is deflected by a block."
            ],
            "PLAYER SUCCESSFUL TACKLE": [
                "A well-timed tackle.",
                "Winning back possession with a clean challenge.",
                "A perfect tackle to regain the ball.",
                "Excellent defensive work with that tackle.",
                "A strong challenge to win the ball."
            ]
        }
        
        # Filler templates for when there's no action
        self.filler_templates = [
            "The teams are battling for possession in midfield.",
            "Both sides looking to establish control of the game.",
            "It's been an interesting contest so far.",
            "The pace of the game has slowed down a bit now.",
            "Players looking to find some space on the pitch.",
            "The managers will be pleased with the effort shown.",
            "What a great atmosphere here today.",
            "Fans on both sides making themselves heard.",
            "We've seen some quality football at times in this match.",
            "The conditions are perfect for football today.",
            "Both teams showing good tactical awareness.",
            "The players are certainly putting in the effort today.",
            "This has been an entertaining encounter.",
            "The tempo of the match has been excellent.",
            "Some good individual battles across the pitch.",
            "The technical quality on display has been impressive.",
            "We're seeing some good movement off the ball.",
            "Both teams are well-organized defensively.",
            "There's a good flow to this game.",
            "The intensity hasn't dropped for a moment."
        ]
        
        # Load player data if CSV path is provided
        self.players = {}
        if self.player_csv_path:
            try:
                self.players = self._load_player_data()
                print(f"Loaded data for {len(self.players)} players")
            except Exception as e:
                print(f"Warning: Could not load player data: {e}")
                print("Continuing with generic commentary")
        
    def _load_player_data(self):
        """Load player data from the CSV file."""
        players = {}
        if not os.path.exists(self.player_csv_path):
            print(f"Warning: Player CSV file {self.player_csv_path} not found!")
            return players
            
        try:
            with open(self.player_csv_path, 'r') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    # Assuming the CSV has columns: jersey_number, name, team
                    try:
                        players[row['jersey_number']] = {
                            'name': row['name'],
                            'team': row['team']
                        }
                    except KeyError as e:
                        print(f"Warning: Missing required column in CSV: {e}")
                        print("Make sure CSV has 'jersey_number', 'name', and 'team' columns")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            
        return players
    
    def _load_events(self):
        """Load events from the JSON file."""
        try:
            with open(self.json_file_path, 'r') as json_file:
                data = json.load(json_file)
                
            # Extract predictions from the data structure
            if "predictions" in data:
                events = data["predictions"]
            else:
                events = data  # Fallback to old format if needed
                
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return []
        
        # Filter events by confidence threshold
        filtered_events = [
            event for event in events 
            if float(event.get('confidence', 0)) >= self.confidence_threshold
        ]
        
        # Sort events by game time
        sorted_events = sorted(filtered_events, key=lambda x: self._parse_game_time(x['gameTime']))
        
        return sorted_events
    
    def _parse_game_time(self, game_time_str):
        """Convert game time string to seconds for sorting."""
        # Remove any extra spaces in the time string
        game_time_str = game_time_str.strip()
        match = re.match(r'(\d+) *- *(\d+):(\d+)', game_time_str)
        if match:
            half, minutes, seconds = map(int, match.groups())
            return (half - 1) * 45 * 60 + minutes * 60 + seconds
        return 0
    
    def _format_time_for_srt(self, seconds):
        """Format time in SRT format: HH:MM:SS,mmm"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        millisecs = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02}:{minutes:02}:{secs:02},{millisecs:03}"
    
    def _get_player_name(self, player_id):
        """Get player name from ID, with fallback to generic terms."""
        if player_id == "NA":
            return "a teammate"
        
        # If we have player mapping data
        player = self.players.get(player_id)
        if player:
            return player['name']
        
        # Fallback: generic player reference
        return f"Number {player_id}"
    
    def _get_team_name(self, team_side):
        """Get team name from side (left/right)."""
        return self.team_names.get(team_side, team_side.capitalize() + " team")
    
    def _get_commentary_for_event(self, event):
        """Generate commentary for a single event."""
        label = event['label']
        player_id = event.get('actor', 'NA')
        secondary_player_id = event.get('secondaryActor', 'NA')
        team_side = event.get('team', 'unknown')
        
        # Get player name and team
        player_name = self._get_player_name(player_id)
        secondary_player_name = self._get_player_name(secondary_player_id)
        team_name = self._get_team_name(team_side)
        
        # Check if we have templates for this label
        if label in self.templates:
            # Decide whether to use specific or generic templates
            if (player_id != 'NA' and player_id in self.players) or player_id == 'NA':
                templates = self.templates[label]
            else:
                # Fall back to generic templates if player isn't in our database
                templates = self.generic_templates.get(label, self.templates[label])
            
            template = random.choice(templates)
            
            # Format the template with player names and team
            try:
                return template.format(
                    player=player_name,
                    secondary_player=secondary_player_name,
                    team_name=team_name
                )
            except KeyError:
                # If template formatting fails, use a generic template
                if label in self.generic_templates:
                    return random.choice(self.generic_templates[label])
                return f"Action from {team_name}."
        
        # Generic commentary for labels without specific templates
        return f"Action from {team_name}."
    
    def _get_filler_commentary(self):
        """Get random filler commentary."""
        return random.choice(self.filler_templates)
    
    def _game_time_to_seconds(self, game_time_str):
        """Convert game time string to seconds since start."""
        # Remove any extra spaces in the time string
        game_time_str = game_time_str.strip()
        match = re.match(r'(\d+) *- *(\d+):(\d+)', game_time_str)
        if match:
            half, minutes, seconds = map(int, match.groups())
            return (half - 1) * 45 * 60 + minutes * 60 + seconds
        return 0
    
    def _check_ffmpeg_available(self):
        """Check if ffmpeg is available on the system."""
        try:
            # Use a platform-independent way to check for ffmpeg
            if os.name == 'nt':  # Windows
                exit_code = os.system('where ffmpeg >nul 2>&1')
            else:  # Unix/Linux/Mac
                exit_code = os.system('which ffmpeg >/dev/null 2>&1')
            return exit_code == 0
        except:
            return False
    
    def generate_srt(self, output_file, duration_seconds=3):
        """
        Generate SRT file with commentary.
        
        Args:
            output_file: Path to the output SRT file
            duration_seconds: Duration in seconds for each subtitle
        """
        events = self._load_events()
        
        if not events:
            print("No events found or JSON file could not be loaded!")
            return False
        
        try:
            with open(output_file, 'w') as srt_file:
                subtitle_count = 1
                last_event_time = None
                
                for i, event in enumerate(events):
                    event_time_seconds = self._game_time_to_seconds(event['gameTime'])
                    
                    # Check if we need to insert filler commentary
                    if last_event_time is not None and event_time_seconds - last_event_time >= 2:
                        # Add filler commentary
                        filler = self._get_filler_commentary()
                        filler_start = last_event_time + 1
                        
                        # Only add filler if it doesn't overlap with next event
                        if filler_start + duration_seconds <= event_time_seconds:
                            srt_file.write(f"{subtitle_count}\n")
                            srt_file.write(f"{self._format_time_for_srt(filler_start)} --> {self._format_time_for_srt(filler_start + duration_seconds)}\n")
                            srt_file.write(f"{filler}\n\n")
                            subtitle_count += 1
                    
                    # Get commentary for current event
                    commentary = self._get_commentary_for_event(event)
                    
                    # Write to SRT file
                    srt_file.write(f"{subtitle_count}\n")
                    srt_file.write(f"{self._format_time_for_srt(event_time_seconds)} --> {self._format_time_for_srt(event_time_seconds + duration_seconds)}\n")
                    srt_file.write(f"{commentary}\n\n")
                    
                    subtitle_count += 1
                    last_event_time = event_time_seconds
            
            print(f"Generated SRT file with {subtitle_count-1} commentary lines")
            return True
            
        except Exception as e:
            print(f"Error generating SRT file: {e}")
            return False
    
    def overlay_on_video(self, video_path, output_path):
        """
        Overlay SRT subtitles on video.
        This requires ffmpeg to be installed on the system.
        
        Args:
            video_path: Path to the input video file
            output_path: Path to the output video file with subtitles
        """
        # Generate SRT with same name as video but in current directory
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        srt_path = f"{video_name}.srt"
        
        # Generate the SRT file
        if not self.generate_srt(srt_path):
            print("Failed to generate SRT file. Cannot overlay on video.")
            return False
        
        # Check if ffmpeg is available
        if not self._check_ffmpeg_available():
            print("ffmpeg does not appear to be installed or is not in PATH.")
            print(f"SRT file has been generated at {srt_path}. You can manually add it to your video.")
            return False
        
        try:
            # Use ffmpeg to overlay subtitles
            cmd = f'ffmpeg -i "{video_path}" -vf subtitles="{srt_path}" "{output_path}"'
            result = os.system(cmd)
            
            if result == 0:
                print(f"Video with commentary overlaid saved to {output_path}")
                return True
            else:
                print("ffmpeg command failed. Please check if the paths are correct.")
                return False
                
        except Exception as e:
            print(f"Error overlaying subtitles on video: {e}")
            return False
            
    def generate_audio_commentary(self, srt_path, output_audio_path="commentary.wav"):
        """Generate audio commentary using system TTS voices with error handling."""
        try:
            # Initialize TTS engine
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            
            # Load and parse SRT file
            if not os.path.exists(srt_path):
                raise FileNotFoundError(f"SRT file not found: {srt_path}")

            with open(srt_path, 'r') as f:
                srt_content = f.read()

            audio_clips = []
            lines = [line.strip() for line in srt_content.split('\n\n') if line.strip()]

            for line in lines:
                parts = line.split('\n')
                if len(parts) < 3:
                    continue
                
                text = parts[2]
                temp_path = "temp_commentary.wav"
                
                try:
                    # Generate audio
                    engine.save_to_file(text, temp_path)
                    engine.runAndWait()
                    
                    # Verify audio file was created
                    if not os.path.exists(temp_path):
                        raise RuntimeError(f"Failed to generate audio for: {text}")
                        
                    # Load and store audio
                    audio_clip = AudioSegment.from_wav(temp_path)
                    audio_clips.append(audio_clip)
                    
                finally:
                    # Cleanup temp file even if errors occur
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            # Combine and export audio
            if audio_clips:
                combined_audio = sum(audio_clips)
                combined_audio.export(output_audio_path, format="wav")
                print(f"Successfully generated audio commentary: {output_audio_path}")
                return output_audio_path
                
            print("No valid commentary lines found in SRT file")
            return None

        except Exception as e:
            print(f"Error generating audio commentary: {str(e)}")
            # Cleanup partial output if exists
            if os.path.exists(output_audio_path):
                os.remove(output_audio_path)
            return None
        
    def overlay_audio_on_video(self, video_path, audio_path, output_path):
        """Add commentary audio to video using ffmpeg command line."""
        try:
            # Verify input files
            if not all(os.path.exists(f) for f in [video_path, audio_path]):
                raise FileNotFoundError("Missing input files")

            # First check if the video has audio
            cmd_check = [
                'ffprobe', 
                '-v', 'error', 
                '-select_streams', 'a', 
                '-show_entries', 'stream=codec_name', 
                '-of', 'default=noprint_wrappers=1:nokey=1', 
                video_path
            ]
            
            result = subprocess.run(cmd_check, capture_output=True, text=True)
            has_audio = result.stdout.strip() != ""
            
            # Choose appropriate ffmpeg command based on whether the video has audio
            if has_audio:
                # If video has audio, mix it with the commentary
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-i', audio_path,
                    '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=longest',
                    '-c:v', 'copy',
                    '-shortest',
                    output_path
                ]
            else:
                # If video has no audio, simply add the commentary as the audio track
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-i', audio_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac',  # Convert audio to AAC for better compatibility
                    '-map', '0:v',  # Map video from first input
                    '-map', '1:a',  # Map audio from second input
                    '-shortest',
                    output_path
                ]
            
            # Run the command with a timeout
            print(f"Running ffmpeg command: {' '.join(cmd)}")
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Check if command was successful
            if process.returncode == 0:
                print(f"Successfully processed: {output_path}")
                return True
            else:
                print(f"Processing failed. Details:\n{process.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("Processing timed out after 5 minutes. The video might be too large or there might be an issue with ffmpeg.")
            return False
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
def find_matching_json(video_name, events_dir="./events_data"):
    """Find the corresponding JSON file for a video name in the events directory."""
    json_path = os.path.join(events_dir, f"{video_name}.json")
    if os.path.exists(json_path):
        return json_path
    
    # If no exact match, list available JSON files
    print(f"No exact match for {video_name}.json in {events_dir}")
    try:
        if not os.path.exists(events_dir):
            print(f"Directory {events_dir} does not exist!")
            return None
            
        json_files = [f for f in os.listdir(events_dir) if f.endswith('.json')]
        if json_files:
            print("Available JSON files:")
            for i, json_file in enumerate(json_files):
                print(f"{i+1}. {json_file}")
            
            choice = input("Enter number of JSON file to use (or press Enter to cancel): ")
            if choice and choice.isdigit() and 1 <= int(choice) <= len(json_files):
                return os.path.join(events_dir, json_files[int(choice)-1])
        else:
            print(f"No JSON files found in {events_dir}")
    except Exception as e:
        print(f"Error listing JSON files: {e}")
    
    return None

def main():
    """Modified usage of the CommentaryGenerator with automatic text and audio processing."""
    # Define directories
    video_dir = "../Input_Videos"
    events_dir = "../events_data"
    output_dir = "../Output_Videos"
    teamsheets_dir = "../teamsheets"
    
    # Ensure directories exist
    for directory in [video_dir, events_dir, output_dir, teamsheets_dir]:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Warning: Could not create directory {directory}: {e}")
    
    # List available videos
    try:
        if not os.path.exists(video_dir):
            print(f"Video directory {video_dir} does not exist!")
            video_path = input(f"Enter full path to video file: ")
            if not os.path.exists(video_path):
                print(f"Error: Video file {video_path} not found! Exiting.")
                return
        else:
            videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            if not videos:
                print(f"No videos found in {video_dir}")
                video_path = input(f"Enter full path to video file: ")
                if not os.path.exists(video_path):
                    print(f"Error: Video file {video_path} not found! Exiting.")
                    return
            else:
                print("Available videos:")
                for i, video in enumerate(videos):
                    print(f"{i+1}. {video}")
                
                choice = input("Enter number of the video to process (or press Enter to cancel): ")
                if not choice or not choice.isdigit() or int(choice) < 1 or int(choice) > len(videos):
                    print("Invalid selection or cancelled. Exiting.")
                    return
                
                video_file = videos[int(choice)-1]
                video_path = os.path.join(video_dir, video_file)
    except Exception as e:
        print(f"Error listing videos: {e}")
        video_path = input(f"Enter full path to video file: ")
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found! Exiting.")
            return
    
    # Get the video name without extension for finding the JSON file
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Find the corresponding JSON file
    json_path = find_matching_json(video_name, events_dir)
    if not json_path:
        json_path = input("Enter the path to the JSON file with event data: ")
        if not os.path.exists(json_path):
            print(f"Error: JSON file {json_path} not found! Exiting.")
            return
    
    # Check for player CSV file with same name as video in teamsheets directory
    player_csv = os.path.join(teamsheets_dir, f"{video_name}.csv")
    if not os.path.exists(player_csv):
        print(f"Warning: Player CSV file {player_csv} not found! Using generic player references.")
        player_csv = None
    else:
        print(f"Found player data file: {player_csv}")
    
    # Get team names
    home_team = input("Enter home team name (left side) or press Enter for default: ")
    away_team = input("Enter away team name (right side) or press Enter for default: ")
    
    team_names = {
        "left": home_team if home_team else "Barnsley",
        "right": away_team if away_team else "Coventry"
    }
    
    print("Initializing commentary generator...")
    # Initialize the commentary generator
    generator = CommentaryGenerator(
        json_file_path=json_path,
        player_csv_path=player_csv,
        team_names=team_names
    )
    
    # Generate SRT file
    print("Generating SRT commentary file...")
    srt_path = f"{video_name}.srt"
    if generator.generate_srt(srt_path):
        print(f"SRT file generated at {srt_path}")
        
        # Step 1: Automatically overlay text commentary on video
        text_output_video = os.path.join(output_dir, f"{video_name}_with_text.mp4")
        print(f"Overlaying text commentary on video and saving to {text_output_video}...")
        text_success = generator.overlay_on_video(video_path, text_output_video)
        
        if text_success:
            # Step 2: Generate audio commentary
            print("Generating audio commentary...")
            audio_path = generator.generate_audio_commentary(srt_path)
            
            if audio_path:
                # Step 3: Add audio to video with text commentary
                final_output_video = os.path.join(output_dir, f"{video_name}_final.mp4")
                print(f"Adding audio commentary to video and saving final output to {final_output_video}...")
                audio_success = generator.overlay_audio_on_video(
                    video_path=text_output_video,  # Use the video with text commentary
                    audio_path=audio_path,
                    output_path=final_output_video
                )
                
                if audio_success:
                    print("\nProcessing complete!")
                    print(f"Final video with both text and audio commentary: {final_output_video}")
                else:
                    print("Failed to add audio commentary to video.")
            else:
                print("Failed to generate audio commentary.")
        else:
            print("Failed to overlay text commentary on video.")
    else:
        print("Failed to generate SRT file.")

if __name__ == "__main__":
    main()

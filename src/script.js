// script.js
// Function to load the videos based on the input video name
function loadVideos() {
    var videoName = document.getElementById('videoName').value.trim();
    if (videoName) {
        // Construct input and output video paths
        //var inputVideoPath = `http://localhost:8000/home/fawwaz/FootballProjectRep/Football_Commentary_System/Football-object-detection/test_videos/${videoName}`;
        //var outputVideoPath = `http://localhost:8000/home/fawwaz/FootballProjectRep/Football_Commentary_System/Football-object-detection/output/${videoName.replace('.mp4', '_out.mp4')}`;
        
        var inputVideoPath = `http://localhost:8000/Football-object-detection/test_videos/${videoName}`;
        var outputVideoPath = `http://localhost:8000/Football-object-detection/output/${videoName.replace('.mp4', '_out.mp4')}`;
        
        
        // Debugging statements to check paths
        console.log("Input Video Path:", inputVideoPath);
        console.log("Output Video Path:", outputVideoPath);
        
        // Create new video elements to check if the sources exist
        var inputVideo = document.createElement('video');
        var outputVideo = document.createElement('video');
        
        // Set the video sources
        inputVideo.src = inputVideoPath;
        outputVideo.src = outputVideoPath;
        
        // Check if the video sources are valid
        if (inputVideo.src && outputVideo.src) {
            console.log("Video sources set, now displaying videos.");
            
            // Set the actual video elements' sources
            document.getElementById('inputVideoSource').src = inputVideoPath;
            document.getElementById('outputVideoSource').src = outputVideoPath;
            
            // Show video containers
            document.getElementById('videoContainer').style.display = "flex";
        } else {
            // Debugging statements for video file not found errors
            if (!inputVideo.src) {
                console.error(`Error: Could not find the input video file at the specified path: ${inputVideoPath}`);
            }
            if (!outputVideo.src) {
                console.error(`Error: Could not find the output video file at the specified path: ${outputVideoPath}`);
            }
            
            alert(`Error: Could not find the video files at the specified paths.\nInput video: ${inputVideoPath}\nOutput video: ${outputVideoPath}`);
            console.log("Video files not found.");
        }
    } else {
        // Debugging statement for no video name entered
        console.log("No video name entered.");
        alert("Please enter a valid video name.");
    }
}

// Play/Pause function for both videos
function playPause() {
    const inputVideo = document.getElementById('inputVideo');
    const outputVideo = document.getElementById('outputVideo');
    
    // Debugging statements for video play/pause
    if (inputVideo.paused) {
        console.log("Playing both videos");
        inputVideo.play();
        outputVideo.play();
    } else {
        console.log("Pausing both videos");
        inputVideo.pause();
        outputVideo.pause();
    }
}

// Sync the output video with the input video
function syncVideos() {
    const inputVideo = document.getElementById('inputVideo');
    const outputVideo = document.getElementById('outputVideo');
    
    // Debugging statement for video synchronization
    console.log("Syncing videos. Input currentTime:", inputVideo.currentTime);
    outputVideo.currentTime = inputVideo.currentTime;
}

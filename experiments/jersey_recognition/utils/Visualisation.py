def draw_keypoints(img, keypoints, scores, threshold=0.3):
    """Draw keypoints on image with confidence threshold"""
    vis_img = img.copy()
    keypoint_names = {
        5: ('Left Shoulder', (0, 0, 255)),    # Red
        6: ('Right Shoulder', (0, 255, 0)),   # Green
        11: ('Left Hip', (255, 0, 0)),        # Blue
        12: ('Right Hip', (255, 255, 0))      # Cyan
    }
    
    for idx, (name, color) in keypoint_names.items():
        if scores[idx] > threshold:
            x, y = map(int, keypoints[idx][:2])
            cv2.circle(vis_img, (x, y), 3, color, -1)
            cv2.putText(vis_img, name, (x-10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return vis_img

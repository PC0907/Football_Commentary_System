import numpy as np
import cv2 as cv

# Initialize the points lists for the source and destination
src_list = []
dst_list = []

def detect_field_keypoints(image):
    """
    Automatically detect keypoints (box corners, center circle) on the football field.
    """
    # Convert to grayscale and use edge detection to identify field lines
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)

    # Detect contours based on edges
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Placeholder logic to detect rectangles (e.g., corners) and the center circle.
    # We would need to refine this based on the specific image details.
    
    keypoints = []
    for cnt in contours:
        # Approximate contour with accuracy proportional to contour perimeter
        epsilon = 0.02 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) == 4:  # Assuming rectangular shapes for field corners
            for point in approx:
                keypoints.append(point[0].tolist())
                if len(keypoints) >= 4:
                    break
        if len(keypoints) >= 4:
            break
    """
    # Detect the center circle (based on circularity)
    # Circle might be deformed into an ellipse in a broadcast view, need to take it into account while rewriting 
    """
    for cnt in contours:
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if 0.7 < circularity < 1.3:  # A circular contour
            M = cv.moments(cnt)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            keypoints.append([cx, cy])
            break

    return keypoints

def get_plan_view(src, dst):
    src_pts = np.array(src_list).reshape(-1, 1, 2)
    dst_pts = np.array(dst_list).reshape(-1, 1, 2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    plan_view = cv.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))
    return plan_view

# Load the source and destination images
# Need to modify this to process frames instead of images
src = cv.imread('imgs/src.jpg')
dst = cv.imread('imgs/dst.jpg')

# Automatically detect keypoints in the src image
src_list = detect_field_keypoints(src)
"""
Prepare a 2D Football Field image/diagram; make a 1080p representation, mark keypoints -> name keypoints ->  note pixel coordinates -> add it to dst_list
"""
dst_list = [[50, 50], [400, 50], [50, 400], [400, 400], [225, 225]]

# Verify detected points and calculate plan view
if len(src_list) == 5 and len(dst_list) == 5:
    plan_view = get_plan_view(src, dst)
    cv.imshow("Plan View", plan_view)

cv.waitKey(0)
cv.destroyAllWindows()


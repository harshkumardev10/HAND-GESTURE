import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def download_model():
    model_path = 'hand_landmarker.task'
    if not os.path.exists(model_path):
        print("Downloading MediaPipe hand tracking model...")
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")
    return model_path

def main():
    model_path = download_model()

    # Create an HandLandmarker object.
    # Set num_hands to 2 to support drawing with both hands at the same time!
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2)
    
    try:
        detector = vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Failed to create HandLandmarker: {e}")
        return

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    # Create a blank canvas to draw on
    canvas = None

    # Fingertips to track
    # 4: Thumb, 8: Index, 12: Middle, 16: Ring, 20: Pinky
    fingertip_ids = [4, 8, 12, 16, 20]
    
    # Colors for each fingertip (BGR format)
    colors = {
        4: (255, 255, 0),  # Cyan for Thumb
        8: (255, 0, 0),    # Blue for Index
        12: (0, 255, 0),   # Green for Middle
        16: (0, 0, 255),   # Red for Ring
        20: (255, 0, 255)  # Magenta for Pinky
    }

    # Store previous points: maps (hand_idx, fingertip_id) -> (x, y)
    prev_points = {}

    print("Starting webcam...")
    print("-> Move any fingertip to draw with different colors!")
    print("-> Press 'c' to clear the canvas.")
    print("-> Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip the frame horizontally for a more intuitive selfie-view
        frame = cv2.flip(frame, 1)
        
        # Initialize canvas if it's None (matches frame dimensions)
        if canvas is None:
            canvas = np.zeros_like(frame)

        # Convert the BGR image from OpenCV to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hand landmarks
        detection_result = detector.detect(mp_image)

        # We will collect the new points for this frame
        current_points = {}

        # If hands are detected
        if detection_result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                h, w, c = frame.shape
                wrist = hand_landmarks[0]
                
                open_fingers = []
                for finger_id in fingertip_ids:
                    # Check if finger is folded
                    if finger_id == 4: # Thumb
                        # Compare thumb tip (4) to pinky base (17) vs thumb IP joint (3) to pinky base
                        base_ref = hand_landmarks[17]
                        dist_tip = math.hypot(hand_landmarks[4].x - base_ref.x, hand_landmarks[4].y - base_ref.y)
                        dist_joint = math.hypot(hand_landmarks[3].x - base_ref.x, hand_landmarks[3].y - base_ref.y)
                        is_open = dist_tip > dist_joint
                    else:
                        # Index (8), Middle (12), Ring (16), Pinky (20)
                        # Compare distance from tip to wrist vs PIP joint to wrist
                        joint = hand_landmarks[finger_id - 2]
                        dist_tip = math.hypot(hand_landmarks[finger_id].x - wrist.x, hand_landmarks[finger_id].y - wrist.y)
                        dist_joint = math.hypot(joint.x - wrist.x, joint.y - wrist.y)
                        is_open = dist_tip > dist_joint
                    
                    if is_open:
                        open_fingers.append(finger_id)
                
                # Because the image is flipped horizontally (mirror), your physical left hand looks like a right hand to the detector
                hand_label = detection_result.handedness[hand_idx][0].category_name
                is_physical_left_hand = (hand_label == "Right")
                
                if is_physical_left_hand:
                    if len(open_fingers) == 0:
                        # Eraser Mode: Use palm center (landmark 9)
                        palm_center = hand_landmarks[9]
                        cx, cy = int(palm_center.x * w), int(palm_center.y * h)
                        
                        # Draw a white circle on screen to indicate eraser mode
                        cv2.circle(frame, (cx, cy), 60, (255, 255, 255), 2)
                        cv2.putText(frame, "LEFT ERASER", (cx - 50, cy - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Erase from canvas (color 0,0,0)
                        cv2.circle(canvas, (cx, cy), 60, (0, 0, 0), cv2.FILLED)
                else:
                    # Drawing Mode (Right hand only)
                    for finger_id in open_fingers:
                        landmark = hand_landmarks[finger_id]
                        curr_x, curr_y = int(landmark.x * w), int(landmark.y * h)
                        
                        # Key for tracking matches hand and specific finger
                        key = (hand_idx, finger_id)
                        current_points[key] = (curr_x, curr_y)

                        color = colors[finger_id]
                        # Draw a circle on the actual finger pointer
                        cv2.circle(frame, (curr_x, curr_y), 8, color, cv2.FILLED)

                        # If we tracked this finger in the last frame, draw a line!
                        if key in prev_points:
                            p_x, p_y = prev_points[key]
                            
                            # Distance check prevents huge lines going across the screen 
                            # if the hand tracking glitches or swaps hands suddenly.
                            distance = math.hypot(curr_x - p_x, curr_y - p_y)
                            if distance < 150: 
                                cv2.line(canvas, (p_x, p_y), (curr_x, curr_y), color, 5)
                            
        # Update our history for the next frame
        prev_points = current_points

        # Combine the original frame and the canvas
        combined = cv2.addWeighted(frame, 1, canvas, 1, 0)

        # Display the result
        cv2.imshow("Hand Gesture Drawing - Multi Pointers", combined)

        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Clear the canvas
            canvas = np.zeros_like(frame)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

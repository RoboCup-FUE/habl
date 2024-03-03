import cv2
import numpy as np

def count_open_fingers(hand_landmarks):
    # Define the landmarks for each finger
    finger_landmarks = {
        "thumb": [4, 3, 2, 1],
        "index": [8, 7, 6, 5],
        "middle": [12, 11, 10, 9],
        "ring": [16, 15, 14, 13],
        "pinky": [20, 19, 18, 17]
    }

    open_fingers = 0

    for finger in finger_landmarks.values():
        # Calculate the distance between the tip of the finger and the base
        length = np.linalg.norm(np.array(hand_landmarks[finger[0]]) - np.array(hand_landmarks[finger[3]]))
        # If the distance is greater than a threshold, consider the finger open
        if length > 30:  # You may need to adjust this threshold based on your setup
            open_fingers += 1

    return open_fingers

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Your code for hand detection and landmark estimation using OpenCV goes here
    # Let's assume you have a variable hand_landmarks which contains the landmarks of the hand

    # Example: hand_landmarks = {"thumb": (x1, y1), "index": (x2, y2), ...}

    # Count the open fingers
    open_fingers = count_open_fingers(hand_landmarks)

    # Determine which hand has more open fingers
    if open_fingers > 0:
        left_count = sum(hand_landmarks["left_hand"])
        right_count = sum(hand_landmarks["right_hand"])

        if left_count > right_count:
            print("left")
        elif right_count > left_count:
            print("right")

    # Display the frame with any overlays or information you need
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

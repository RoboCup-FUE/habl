import cv2
import mediapipe as mp
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

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image with MediaPipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark positions
            hand_landmarks_dict = {}
            for idx, landmark in enumerate(hand_landmarks.landmark):
                hand_landmarks_dict[idx] = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))

            # Count the open fingers
            open_fingers = count_open_fingers(hand_landmarks_dict)

            # Display the number of open fingers
            cv2.putText(frame, f"Open Fingers: {open_fingers}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Determine which hand has more open fingers
            if open_fingers > 0:
                # Assuming the first hand detected is the left hand and the second one is the right hand
                if len(results.multi_hand_landmarks) == 1:
                    print("left")
                elif len(results.multi_hand_landmarks) == 2:
                    print("right")

            # Draw landmarks on the image
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

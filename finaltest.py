import mediapipe as mp
import cv2
import numpy as np
import pyautogui
import win32api
import time
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

video = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    click_threshold = 30  # distance threshold to trigger click event
    prev_click_position = None
    click_cooldown = 1  # cooldown time in seconds
    last_click_time = time.time() - click_cooldown

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image_height, image_width, _ = image.shape

        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

                for landmark in mp_hands.HandLandmark:
                    if landmark == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                        normalized_landmark = hand.landmark[landmark]
                        pixel_coordinates = mp_drawing._normalized_to_pixel_coordinates(
                            normalized_landmark.x, normalized_landmark.y, image_width, image_height)
                        if pixel_coordinates:
                            cv2.circle(image, pixel_coordinates, 25, (0, 200, 0), 5)

                            index_finger_tip_x = pixel_coordinates[0]
                            index_finger_tip_y = pixel_coordinates[1]

                            # Adjust cursor position
                            scaled_x = int(index_finger_tip_x * 4)
                            scaled_y = int(index_finger_tip_y * 5)
                            win32api.SetCursorPos((scaled_x, scaled_y))

                            # Check if we should perform a click
                            current_time = time.time()
                            if prev_click_position:
                                dist = np.linalg.norm(np.array([index_finger_tip_x, index_finger_tip_y]) - np.array(prev_click_position))
                                if dist < click_threshold and (current_time - last_click_time) > click_cooldown:
                                    pyautogui.click()
                                    last_click_time = current_time

                            prev_click_position = (index_finger_tip_x, index_finger_tip_y)

        cv2.imshow("Hand Tracking", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()

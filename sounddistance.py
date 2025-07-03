import cv2
import mediapipe as mp
import math
import pygame
from pythonosc.udp_client import SimpleUDPClient

# Configura OSC
ip = "127.0.0.1"
porta = 8000
client = SimpleUDPClient(ip, porta)

# Inizializza MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Inizializza Pygame mixer e carica i suoni
pygame.mixer.init()
sound_tamburo = pygame.mixer.Sound("tamburo1.wav")   # Mano destra
sound_maracas = pygame.mixer.Sound("maracas.wav")    # Mano sinistra

# Avvia la webcam
cap = cv2.VideoCapture(0)

prev_state = {}

def distanza(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Right" o "Left"
            hand_id = label.lower()  # 'right' o 'left'

            lm_list = []
            h, w, _ = img.shape
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            if len(lm_list) >= 9:
                thumb_tip = lm_list[4]
                index_tip = lm_list[8]

                cv2.circle(img, thumb_tip, 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, index_tip, 10, (0, 255, 0), cv2.FILLED)
                cv2.line(img, thumb_tip, index_tip, (0, 255, 255), 3)

                dist = int(distanza(thumb_tip, index_tip))
                cv2.putText(img, f'{label} hand - Distanza: {dist}px', (10, 50 if hand_id == 'right' else 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                threshold = 50

                if dist <= threshold and prev_state.get(hand_id) != "closed":
                    # Suona suono a seconda della mano
                    if hand_id == "right":
                        sound_tamburo.play()
                    elif hand_id == "left":
                        sound_maracas.play()

                    prev_state[hand_id] = "closed"
                    client.send_message(f"/mano/{hand_id}/stato", 0)
                    client.send_message(f"/mano/{hand_id}/distanza", dist)

                elif dist > threshold and prev_state.get(hand_id) != "open":
                    prev_state[hand_id] = "open"
                    client.send_message(f"/mano/{hand_id}/stato", 1)
                    client.send_message(f"/mano/{hand_id}/distanza", dist)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Tracking mani - Tamburo e Maracas", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

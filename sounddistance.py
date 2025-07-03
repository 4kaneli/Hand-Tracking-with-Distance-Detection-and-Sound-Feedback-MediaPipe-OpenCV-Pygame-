import cv2
import mediapipe as mp
import math
import time
import pygame
from pythonosc.udp_client import SimpleUDPClient

# Configura OSC
ip = "127.0.0.1"
porta = 8000
client = SimpleUDPClient(ip, porta)

# Inizializza MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Inizializza Pygame mixer e carica il suono
pygame.mixer.init()
sound_synth = pygame.mixer.Sound("tamburo1.wav")  # suono quando dita chiuse

# Avvia la webcam
cap = cv2.VideoCapture(0)

prev_state = None

def distanza(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            h, w, _ = img.shape
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            if len(lm_list) >= 9:
                thumb_tip = lm_list[4]
                index_tip = lm_list[8]

                cv2.circle(img, thumb_tip, 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, index_tip, 10, (0, 255, 0), cv2.FILLED)
                cv2.line(img, thumb_tip, index_tip, (0, 255, 255), 3)

                dist = int(distanza(thumb_tip, index_tip))
                cv2.putText(img, f'Distanza: {dist}px', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                threshold = 50

                if dist <= threshold and prev_state != "closed":
                    sound_synth.play()  # suono singolo quando chiudi dita
                    prev_state = "closed"
                    client.send_message("/mano/stato", 0)
                    client.send_message("/mano/distanza", dist)
                elif dist > threshold and prev_state != "open":
                    prev_state = "open"
                    client.send_message("/mano/stato", 1)
                    client.send_message("/mano/distanza", dist)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Tracking mano - Suono chiusura", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()


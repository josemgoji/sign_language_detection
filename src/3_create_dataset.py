import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3,max_num_hands=1)

DATA_DIR = '../data_aumt'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []
        
        path = os.path.join(DATA_DIR, dir_, img_path)
        img_rgb = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        
        H, W, _ = img_rgb.shape

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)
                
            min_x = min(x_)
            min_y = min(y_)
            max_x = max(x_)
            max_y = max(y_)
            
            width = max_x - min_x if (max_x - min_x) != 0 else 1e-6
            height = max_y - min_y if (max_y - min_y) != 0 else 1e-6
    
            for i in range(len(x_)):
                            # Normaliza por el bounding box
                            norm_x = (x_[i] - min_x) / width
                            norm_y = (y_[i] - min_y) / height
                            data_aux.append(norm_x)
                            data_aux.append(norm_y)

            data.append(data_aux)
            labels.append(dir_)

f = open('../model/data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
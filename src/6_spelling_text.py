# Parámetros de control
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Cargar modelo
model_dict = pickle.load(open('../model/model.p', 'rb'))
model = model_dict['model']
label_encoder = model_dict['label_encoder']

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configuración correcta para video en tiempo real
hands = mp_hands.Hands(
    static_image_mode=False,  # Cambiado a False para video
    max_num_hands=1,
    min_detection_confidence=0.7,  # Aumentado para mejor detección
    min_tracking_confidence=0.5    # Añadido para mejor tracking
)

history_size = 60  # Tamaño de la ventana de suavizado
min_repeats = 50   # Cuántas veces debe aparecer la letra en la ventana para aceptarla

prediction_history = []
output_text = ""
last_copied = ""  # Última letra copiada

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extraer coordenadas (una sola pasada)
            data_aux = []
            x_ = []
            y_ = []

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Validar que tenemos suficientes landmarks
            if len(x_) < 21:  # MediaPipe devuelve 21 landmarks por mano
                continue

            # Normalización
            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)

            width = max_x - min_x if (max_x - min_x) > 1e-6 else 1e-6
            height = max_y - min_y if (max_y - min_y) > 1e-6 else 1e-6

            for i in range(len(x_)):
                norm_x = (x_[i] - min_x) / width
                norm_y = (y_[i] - min_y) / height
                data_aux.append(norm_x)
                data_aux.append(norm_y)

            # Validar dimensiones para el modelo
            if len(data_aux) != 42:  # 21 landmarks * 2 coordenadas
                continue

            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = label_encoder.inverse_transform(prediction)[0]

                # Suavizado de predicciones
                prediction_history.append(predicted_character)
                if len(prediction_history) > history_size:
                    prediction_history.pop(0)

                # Usar la predicción más frecuente
                final_prediction = max(set(prediction_history), key=prediction_history.count)

                # --- LÓGICA DE COPIADO ---
                # Contar cuántas veces aparece la predicción en la ventana
                count = prediction_history.count(final_prediction)
                # Si es espacio, lo tratamos diferente
                if final_prediction.lower() == "space":
                    if last_copied != " " and count >= min_repeats:
                        output_text += " "
                        last_copied = " "
                        print(f"Texto actual: '{output_text}'")
                else:
                    if final_prediction != last_copied and count >= min_repeats:
                        output_text += final_prediction
                        last_copied = final_prediction
                        print(f"Texto actual: '{output_text}'")
                # --- FIN LÓGICA DE COPIADO ---
                # Bounding box corregido
                x1 = max(0, int(min_x * W) - 10)
                y1 = max(0, int(min_y * H) - 10)
                x2 = min(W, int(max_x * W) + 10)  # Corregido: + en lugar de -
                y2 = min(H, int(max_y * H) + 10)  # Corregido: + en lugar de -

                # Dibujar resultado
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, final_prediction, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                # Mostrar confianza (opcional)
                confidence = max(model.predict_proba([np.asarray(data_aux)])[0])
                cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y2 + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


            except Exception as e:
                print(f"Error en predicción: {e}")
                continue
        
    # Mostrar el texto acumulado en la parte inferior izquierda de la ventana
    margin_x = 20  # margen izquierdo
    margin_y = 20  # margen inferior
    dy = 40        # salto de línea
    max_chars_per_line = 30  # ajusta según el ancho de tu ventana

    # Divide el texto en líneas si es muy largo
    lines = [output_text[i:i+max_chars_per_line] for i in range(0, len(output_text), max_chars_per_line)]
    num_lines = len(lines)

    # Calcula la posición inicial y0 para la primera línea (de abajo hacia arriba)
    H, W, _ = frame.shape
    y0 = H - margin_y - (num_lines - 1) * dy

    for i, line in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(frame, line, (margin_x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import joblib
import cv2
import mediapipe as mp
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

#rna = joblib.load('rede_neural_treinada.pkl')
rna = joblib.load('knn_treinado.pkl')

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # supondo que a variável "hand_landmarks" contém as coordenadas dos pontos de interesse do sinal de libras
            hand_landmarks = [landmark for landmark in hand_landmarks.landmark]
            x_coords = []
            y_coords = []

            for landmark in hand_landmarks[:-1]:
                x_coords.append(landmark.x)
                y_coords.append(landmark.y)
           
            features = np.concatenate((x_coords, y_coords))
          
            predicao = rna.predict(np.reshape(features, (1, -1)))
            # Obter a letra prevista
            predicao_letra = predicao[0]

            # Desenhar o texto na imagem
            texto = "Letra prevista: {}".format(predicao_letra)
            cv2.putText(image, texto, (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            #print("O sinal de libras capturado é a letra:", predicao)

        cv2.imshow('MediaPipe Hands - Letra prevista', image)
        #cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()



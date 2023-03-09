import cv2
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from coletaArquivosDataset import image_list

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.1)


IMAGE_FILES = ["A1.jpg", "B1.jpg"]


# Função para aumentar brilho da imagem
def increase_brightness(img, value=25):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img

colunas=['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP',
                             'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP',
                            'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'] 
# teste = pd.DataFrame(columns=['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP',
#                             'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP',
#                             'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'])
teste = pd.DataFrame()
labels = []
for idx, file in enumerate(image_list):
    print(idx)
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    
    label = str(file[8])
    print(label)
    #image = increase_brightness(img)
    #plt.figure(figsize = [5, 5])

    # Here we will display the sample image as the output.
    #plt.title("Imagem de Entrada");plt.axis('off');plt.imshow(image[:,:,::-1]);plt.show()

    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Initially set finger count to 0 for each cap
    fingerCount = 0

    if not results.multi_hand_landmarks:
        image = increase_brightness(image)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand index to check label (left or right)
            handIndex = results.multi_hand_landmarks.index(hand_landmarks)
            handLabel = results.multi_handedness[handIndex].classification[0].label

        # Set variable to keep landmarks positions (x and y)
            handLandmarks = []

        # Fill list with x and y positions of each landmark
            for landmarks in hand_landmarks.landmark:
                handLandmarks.append([landmarks.x, landmarks.y])

            # Test conditions for each finger: Count is increased if finger is 
            #   considered raised.
            # Thumb: TIP x position must be greater or lower than IP x position, 
            #   deppeding on hand label.
            if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                fingerCount = fingerCount+1
            elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                fingerCount = fingerCount+1

            # Other fingers: TIP y position must be lower than PIP y position, 
            #   as image origin is in the upper left corner.
            if handLandmarks[8][1] < handLandmarks[6][1]:       #Index finger
                fingerCount = fingerCount+1
            if handLandmarks[12][1] < handLandmarks[10][1]:     #Middle finger
                fingerCount = fingerCount+1
            if handLandmarks[16][1] < handLandmarks[14][1]:     #Ring finger
                fingerCount = fingerCount+1
            if handLandmarks[20][1] < handLandmarks[18][1]:     #Pinky
                fingerCount = fingerCount+1
            
            img_copy = image.copy()
            # Draw hand landmarks 
            mp_drawing.draw_landmarks(
                img_copy,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # arquivo = open('featuresImg' + str(idx) + '.txt', 'w')
        # Posição x,y de algum ponto da mão: handLandmarks[valor]
        valueList = []
        for i in range(21):
            valueList.append(handLandmarks[i])
            #valueListY.append(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y)
            #teste.loc[mp_hands.HandLandmark(i).name] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value]
            #teste = teste.append({mp_hands.HandLandmark(i).name : hand_landmarks.landmark[mp_hands.HandLandmark(i).value]}, ignore_index=True)
            #teste[mp_hands.HandLandmark(i).name] = [hand_landmarks.landmark[mp_hands.HandLandmark(i).value]]
            # teste.insert(i, str(mp_hands.HandLandmark(i).name), 
            #                 str(hand_landmarks.landmark[mp_hands.HandLandmark(i).value]), True)
                            
            # arquivo.write(str(mp_hands.HandLandmark(i).name) 
            #                 + '\n' 
            #                 + str(hand_landmarks.landmark[mp_hands.HandLandmark(i).value]) 
            #                 + '\n\n')
        valueList.append(label)
        teste = teste.append(pd.DataFrame([valueList], columns=['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP',
                             'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP',
                            'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP', 'LABEL']), ignore_index=True)
                            
        #fig = plt.figure(figsize = [5, 5])
    
        #plt.title("Dedos Levantados: " + str(fingerCount));plt.axis('off');plt.imshow(img_copy[:,:,::-1]);plt.show()

#display(teste.head())

teste.to_csv('data2.csv', index=False)

#arquivo.close()


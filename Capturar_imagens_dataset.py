
# import the necessary packages
from Alinhando_faces import FaceAligner
from helpers import rect_to_bb
import imutils
import dlib
import cv2
from imutils import paths
import os
import uuid
import argparse
# Estabelecendo uma conexão com a webcam
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=400)

# Criando os caminhos para as pastas
POS_PATH = os.path.join('data', 'positive') # imagens positivas cadastradas pelo usuário
NEG_PATH = os.path.join('data', 'negative') # iamgens do Dataset de diversas pessoas
ANC_PATH = os.path.join('data', 'anchor') # imagem do usuário recolhida da webcam


cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()
   
    # Se a tecla 'A' for acionada, armazena uma captura do frame
    # na pasta referente ao novo usuário cadastrado
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # Cria um caminho de arquivo exclusivo para cada imagem captada
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        image = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Nesse caso só detecção de uma pessoa
        rects = detector(gray, 2)

        # Extrai o ROI (região de interesse) da face *original*,
        # depois alinha a face usando pontos de referência faciais (landmarks)

        # Loop sobre as detecções de rosto
        for rect in rects:
            (x, y, w, h) = rect_to_bb(rect)
            faceOrig = imutils.resize(image, width=300)
            faceAligned = fa.align(image, gray, rect)

            # Armazena a imagem já com a face alinhada,
            # para facilitar o reconhecimento facial posteriormente
            cv2.imwrite(imgname, faceAligned)
            print('Imagem capturada!')
        # Mostrar imagem de volta à tela
    cv2.imshow('Imagem', frame)

    # Se a tecla 'P' for acionada, armazena uma captura do frame
    # na pasta referente ao novo usuário cadastrado
    if cv2.waitKey(1) & 0XFF == ord('p'):
        # Cria um caminho de arquivo exclusivo para cada imagem captada
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        image = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Nesse caso só detecção de uma pessoa
        rects = detector(gray, 2)

        # Extrai o ROI (região de interesse) da face *original*,
        # depois alinha a face usando pontos de referência faciais (landmarks)

        # Loop sobre as detecções de rosto
        for rect in rects:
            (x, y, w, h) = rect_to_bb(rect)
            faceOrig = imutils.resize(image, width=300)
            faceAligned = fa.align(image, gray, rect)

            # Armazena a imagem já com a face alinhada,
            # para facilitar o reconhecimento facial posteriormente
            cv2.imwrite(imgname, faceAligned)
            print('Imagem capturada!')
        # Mostrar imagem de volta à tela
    cv2.imshow('Imagem', frame)    
    

    # Parar a captura de imagens
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
        
# Soltando a webcam
cap.release()
# Fechando a imagem dos frames
cv2.destroyAllWindows()
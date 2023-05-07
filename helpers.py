
'''
    Código para auxiliar o código 'Alinhando_faces.py',
    contendo as funções implementadas neste código
    e possíveis utilizações futuras.
'''

# Importando bibliotecas necessárias
from collections import OrderedDict
import numpy as np
import cv2

# Define um dicionário que mapeia os índices dos
# pontos de referência facial (landmarks) para regiões específicas do rosto

# Para o detector de 'landmarks' de 68 pontos da dlib:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

# Para o detector de 'landmarks' de 5 pontos da dlib:
FACIAL_LANDMARKS_5_IDXS = OrderedDict([
	("right_eye", (2, 3)),
	("left_eye", (0, 1)),
	("nose", (4))
])

# Para dar suporte ao código 'legacy', padronizamos os índices para o
# modelo de 68 pontos
FACIAL_LANDMARKS_IDXS = FACIAL_LANDMARKS_68_IDXS

def rect_to_bb(rect):
	# Pega o retângulo referente à face previsto por dlib e converte
	# para o formato (x, y, w, h) como seria feito com OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# Retornar uma tupla de (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# Inicializa a lista de coordenadas (x, y)
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)


	# Faz um loop sobre todos os landmarks da face e converte-os
	# para uma tupla de 2 coordenadas (x, y)
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# Retorna a lista de coordenadas (x, y)
	return coords


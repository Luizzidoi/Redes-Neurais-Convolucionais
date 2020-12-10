"""
Redes Neurais Convolucionais
Base de dados do Homer e Bart (pasta 'data_personagens')
Código: 1 - Treinamento;
        2 - Carrega uma imagem da base de dados e informa se é uma imagem do Bart ou do Homer
Obs.: Não é necessário carregar as imagens manualmente utilizando essa classe (ImageDataGenerator)
"""


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# Função utilizada para normalização dos mapas de caracteristicas. Melhorias da rede neural
from keras.layers.normalization import BatchNormalization
# Classe geradora do Augumentation
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
# Faz a leitura de uma imagem
from keras.preprocessing import image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


""" Estrutura de treinamento da rede neural """
classificador = Sequential()
classificador.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2, 2)))

classificador.add(Conv2D(32, (3, 3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2, 2)))

classificador.add(Flatten())

classificador.add(Dense(units=4, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=4, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=1, activation='sigmoid'))

classificador.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


""" Gerar as imagens utilizadas para o treinamento, normalizá-las e redimensioná-las """
gerador_treinamento = ImageDataGenerator(rescale=1./255, rotation_range=7, horizontal_flip=True,
                                         shear_range=0.2, height_shift_range=0.07, zoom_range=0.2)
gerador_teste = ImageDataGenerator(rescale=1./255)


base_treinamento = gerador_treinamento.flow_from_directory('dataset_personagens/training_set', target_size=(64, 64),
                                                           batch_size=10, class_mode='binary')
base_teste = gerador_teste.flow_from_directory('dataset_personagens/test_set', target_size=(64, 64),
                                               batch_size=10, class_mode='binary')


classificador.fit_generator(base_treinamento, steps_per_epoch=196, epochs=100,
                            validation_data=base_teste, validation_steps=73)



""" Previsão de uma imagem qualquer da base de dados """

# Carregar uma imagem para teste
imagem_teste = image.load_img('dataset_personagens/test_set/bart/bart1.bmp', target_size=(64, 64))
# Transforma a imagem para uma matriz de dados dos pixels
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255
# Expande as dimensões da imagem (matriz) pq é a forma que o Tensorflow trabalha com as imagens. Axis = 0 para adc uma coluna
imagem_teste = np.expand_dims(imagem_teste, axis=0)

previsao = classificador.predict(imagem_teste)
# Como é uma função sigmoid, o valor da previsao é uma probabilidade
# Mostra qual o valor de cada classe. Homer = 1 e Bart = 0
print(base_treinamento.class_indices)
print(previsao)
if previsao > 0.5:
    print("A imagem é o Homer!")
else:
    print("A imagem é o Bart!")

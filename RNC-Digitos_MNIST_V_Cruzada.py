"""
Redes Neurais Convolucionais
Base de dados MNIST (Digitos)
Método da Validação Cruzada
Um método diferente do método utilizado anteriormente (com base de dados de treinamento e teste) para analise da melhor precisão
"""


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
import numpy as np
# Importação da classe para a validação cruzada
from sklearn.model_selection import StratifiedKFold
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Validação cruzada feita manualmente na divisão da base de dados. Seed serve para mudar a semente geradora dos números aleatórios
seed = 5
np.random.seed(seed)

(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

# Transformação nos dados para que o Tensorflow consiga fazer a leitura
previsores = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)
previsores = previsores.astype('float32')
# Técnica do min max normalization = normalização dos dados para diminuir a escala
previsores /= 255
# Modificação das classe para tipo dummy
classe = np_utils.to_categorical(y_treinamento, 10)

# Variável kfold para controlar a validação cruzada
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
resultados = []

# Cria uma matriz 5x1 preenchida com zeros
a = np.zeros(5)
# Cria uma matriz 60000 (tamanho da matriz classe) x 1 preenchida com zeros
b = np.zeros(shape=(classe.shape[0], 1))

""" Um formato diferente para validação cruzada """
for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape=(classe.shape[0], 1))):
    # print('Indice treinamento: ', indice_treinamento, 'Indice teste: ', indice_teste)
    classificador = Sequential()
    classificador.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    classificador.add(MaxPooling2D(pool_size=(2, 2)))
    classificador.add(Flatten())

    classificador.add(Dense(units=128, activation='relu'))
    classificador.add(Dense(units=10, activation='softmax'))
    classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento], batch_size=128, epochs=5)

    precisao = classificador.evaluate(previsores[indice_teste], classe[indice_teste])
    resultados.append(precisao[1])

print(resultados)

""" Média dos 6 resultados de precisão encontrado """
media = sum(resultados) / len(resultados)
print(media)
"""
Redes Neurais Convolucionais
Base de dados de Homer e Bart (arquivo '.csv')
Utilizou-se rede neural densa
Código: Faz o treinamento da rede neural e informa a precisão da mesma
Obs.: necessário carregar as imagens manualmente (.csv)
"""


import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


""" Carregar os atributos da base de dados para as variáveis """
base = pd.read_csv('personagens.csv')
previsores = base.iloc[:, 0:6].values
classe = base.iloc[:, 6].values


label_encoder = LabelEncoder()
classe = label_encoder.fit_transform(classe)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)


""" Estrutura de treinamento da Rede Neural """
classificador = Sequential()
# Units = (entradas + saídas) / 2 = 6 + 1 / 2 = 3.5
classificador.add(Dense(units=4, activation='relu', input_dim=6))
classificador.add(Dense(units=4, activation='relu'))
classificador.add(Dense(units=1, activation='sigmoid'))

classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=2000)

""" Printa a precisão da rede """
resultado = classificador.evaluate(previsores_teste, classe_teste)
print(resultado)
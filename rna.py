import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#from tensorflow import keras


# data = pd.read_csv('coordinates.csv')
# dataForm = pd.DataFrame()
# for i in range(20):
#     dataForm[[data.iloc[:, i].name+'_X', data.iloc[:, i].name+'_Y']] = data[data.iloc[:, i].name].apply(lambda x: pd.Series(str(x).replace('[', '').replace(']', '').replace('\'', '').split(',')))
# dataForm["LABEL"] = data["LABEL"]
# dataForm.to_csv('dataForm.csv', index=False)

# Carregar dados do arquivo CSV
data = pd.read_csv('dataForm.csv')


# Converter dataframe para array NumPy
dataset = np.array(data)

# Separar as entradas (X) e saídas (y)
x = dataset[:, 0:40] # seleciona as colunas 0 até 20 (21 não incluso)
y = dataset[:, 40] # seleciona a última coluna (coluna 21)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=10) 


classifier = MLPClassifier(solver='sgd', hidden_layer_sizes=(25,), learning_rate_init=0.01, activation='logistic', max_iter=1500, random_state = 1)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print(classification_report(y_test, y_pred))
print('Acurácia: ', accuracy_score(y_test, y_pred))



# Teste de sample para rodar em tempo real
print('x_test: ', x_test[2])
y_pred = classifier.predict(np.reshape(x_test[2], (1, -1)))
print('predict: ', y_pred)

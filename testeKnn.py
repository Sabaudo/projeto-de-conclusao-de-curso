import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import plotly.express as px


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

# sufInd = np.arange(70932)
# np.random.shuffle(sufInd)

# x_train = x[sufInd[:49652], :]
# x_test = x[sufInd[49652:], :]
# y_train = y[sufInd[:49652]]
# y_test = y[sufInd[49652:]]

classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

print(classification_report(y_test, y_pred))
print('Acurácia: ', accuracy_score(y_test, y_pred))
fig = px.scatter(dataset, x=8, y=9, color=40, labels=["Polegar X", "Polegar Y"])
fig.show()


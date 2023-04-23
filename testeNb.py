import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


# Carregar dados do arquivo CSV
data = pd.read_csv('dataForm.csv')

# Converter dataframe para array NumPy
dataset = np.array(data)

# Separar as entradas (X) e saídas (y)
x = dataset[:, 0:40] # seleciona as colunas 0 até 20 (21 não incluso)
y = dataset[:, 40] # seleciona a última coluna (coluna 21)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1) 

clf = GaussianNB()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)

print('Acurácia naive bayes: ', acc)

print(classification_report(y_test, y_pred))

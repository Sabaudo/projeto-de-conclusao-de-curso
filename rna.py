import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

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
print('feats: ', x.shape)
sys.exit()
classes = np.unique(y)
classes = sorted(classes)

y_filtrado = np.zeros_like(y, dtype=int)
for i, letra in enumerate(classes):
    y_filtrado[y == letra] = i

# Imprime o array de rótulos filtrados
#print(y_filtrado)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=10) 

mlp = MLPClassifier(solver='sgd', hidden_layer_sizes=(15,), learning_rate_init=0.04, activation='logistic', max_iter=1180, random_state = 1)
mlp.fit(x_train, y_train)

joblib.dump(mlp, 'rede_neural_treinada.pkl')

sys.exit()
max_iters = [100, 500, 800, 1000, 1180]
accuracy_list = []
cross_entropies = []

for max_iter in max_iters:
    print('Iteração: ', max_iter)
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(15,), learning_rate_init=0.04, activation='logistic', max_iter=max_iter, random_state = 1)
    clf.fit(x_train, y_train)
    accuracy_list.append(clf.score(x_test, y_test))
    cross_entropies.append(clf.loss_curve_)
    print('acuracia - iteracao ', max_iter,': ',  clf.score(x_test, y_test))
    #print('cross-entropy - iteracao ', max_iter,': ',  clf.loss_curve_)
    y_pred = clf.predict(x_test)
    

# Criando o grid de subplots
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

# Plotando o gráfico de acurácias no primeiro subplot
axs[0].plot(max_iters, accuracy_list, '-o')
axs[0].set_xlabel('Número de iterações')
axs[0].set_ylabel('Acurácia')
axs[0].set_title('Acurácia em função do número de iterações')
axs[0].grid(True)
fig.subplots_adjust(hspace=2.0)

# Plotando o gráfico de cross-entropy no segundo subplot
for i, max_iter in enumerate(max_iters):
    axs[1].plot(cross_entropies[i], label=f"max_iter={max_iter}")
axs[1].set_xlabel('Número de iterações')
axs[1].set_ylabel('Cross-entropy')
axs[1].set_title('Cross-entropy em função do número de iterações')
axs[1].grid(True)

# Ajustando o layout dos subplots
plt.tight_layout()

# Exibindo o gráfico
plt.show()



# padrao: camadas ocultas 25 e iteracoes 1500
classifier = MLPClassifier(solver='sgd', hidden_layer_sizes=(15,), learning_rate_init=0.01, activation='logistic', max_iter=1180, random_state = 1)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
report = classification_report(y_test, y_pred, output_dict=True)

# Extrai as acurácias de cada classe
accuracies = []
for i in range(26):
    if chr(i + 65) in report:
        accuracies.append(report[chr(i + 65)]['precision'])
    else:
        accuracies.append(0)

# Plota o gráfico de barras das acurácias
plt.bar([chr(i + 65) for i in range(26)], accuracies)
plt.title('Acurácias para cada classe')
plt.xlabel('Classe')
plt.ylabel('Acurácia')
plt.ylim([0, 1])
plt.show()


print(classification_report(y_test, y_pred))
print('Acurácia: ', accuracy_score(y_test, y_pred))

fig, ax = plt.subplots(figsize=(15,15))
cm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=classifier.classes_)
cm.plot(ax=ax)
plt.grid(False)
plt.show()

# train_sizes, train_scores, test_scores = learning_curve(
#     estimator=classifier, X=x, y=y, cv=10, scoring='accuracy')
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)

# plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
# plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
# plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
# plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

# plt.xlabel('Number of training samples')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.ylim([0.5, 1.0])
# plt.show()




# Teste de sample para rodar em tempo real
# print('x_test: ', x_test[2])
# y_pred = classifier.predict(np.reshape(x_test[2], (1, -1)))
# print('predict: ', y_pred)

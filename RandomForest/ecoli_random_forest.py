import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from tqdm import tqdm

# Carregar o conjunto de dados usando o módulo csv
file_path = 'ecoli_stats_2020.csv'  # Atualizar com o caminho real
with open(file_path, newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=';')
    header = next(datareader)
    data = [row for row in tqdm(datareader, desc="Loading data")]

# Selecionar características relevantes e a variável alvo
feature_indices = [header.index(col) for col in ['latitude', 'longitude', 'ano', 'nuecoli', 'ambiente']]
target_index = header.index('medecoli')

# Extrair características e dados de alvo, substituindo vírgulas por pontos para conversão numérica
X = [[float(row[i].replace(',', '.')) for i in feature_indices] for row in tqdm(data, desc="Processing features")]
y = [float(row[target_index].replace(',', '.')) for row in tqdm(data, desc="Processing target")]

# Classificar a variável alvo com base nas faixas fornecidas para concentração de E. coli
def classify_ecoli(value):
    if value <= 200:
        return 1  # Excelente
    elif value <= 400:
        return 2  # Muito boa 
    elif value <= 800:
        return 3  # Satisfatória
    else:
        return 4  # Insatisfatória

y_classified = [classify_ecoli(value) for value in tqdm(y, desc="Classifying target")]

# Verificar se y_classified contém apenas rótulos de classe discretos
unique_classes = set(y_classified)
print(f"Unique classes in y_classified: {unique_classes}")

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_classified, test_size=0.2, random_state=42)

# Padronizar as características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir a grade de parâmetros para o GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Inicializar o classificador Random Forest
rf_classifier = RandomForestClassifier(random_state=42)

# Realizar GridSearchCV para encontrar os melhores hiperparâmetros
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Obter o melhor modelo
best_rf_classifier = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# Prever no conjunto de teste usando o melhor modelo
y_pred = best_rf_classifier.predict(X_test_scaled)

# Avaliar o modelo
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Calcular métricas adicionais
FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
TP = np.diag(conf_matrix)
TN = conf_matrix.sum() - (FP + FN + TP)
FPR = FP / (FP + TN)
specificity = TN / (TN + FP)
error_rate = 1 - accuracy

# Imprimir todas as métricas
print(f"Acurácia: {accuracy:.2f}")
print(f"Taxa de erro: {error_rate:.2f}")
print(f"Precisão: {precision:.2f}")
print(f"Especificidade: {np.mean(specificity):.2f}")
print(f"Sensibilidade (Recall): {recall:.2f}")
print(f"FPR: {np.mean(FPR):.2f}")
print(f"F1 Score: {f1:.2f}")

# Plotar a matriz de confusão para o melhor modelo
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Excelente', 'Muito Boa', 'Satisfatória', 'Insatisfatória'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão para \'Escherichia Coli\'')
plt.show()

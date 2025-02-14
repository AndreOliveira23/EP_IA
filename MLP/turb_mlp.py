import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from tqdm import tqdm

# Load the dataset using csv module
file_path = 'turb_stats_2020.csv'  # Update with the actual path
with open(file_path, newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=';')
    header = next(datareader)
    data = [row for row in tqdm(datareader, desc="Loading data")]

# Select relevant features and target variable
feature_indices = [header.index(col) for col in ['latitude', 'longitude', 'ano', 'nuturb', 'ambiente']]
target_index = header.index('medturb')

# Extract features and target data, replacing commas with periods for numeric conversion
X = [[float(row[i].replace(',', '.')) for i in feature_indices] for row in tqdm(data, desc="Processing features")]
y = [float(row[target_index].replace(',', '.')) for row in tqdm(data, desc="Processing target")]

# Classify target variable based on the given ranges for turbidity concentration
def classify_turbidity(value):
    if value <= 5:
        return 1  # Distribuíveis
    elif value <= 40:
        return 2  # Classe 1 ou 2 (Low Turb)
    elif value <= 100:
        return 3  # Classe 3 ou 4 (Moderate Turb)
    else:
        return 4  # High Turb

y_classified = [classify_turbidity(value) for value in tqdm(y, desc="Classifying target")]

# Verify that y_classified contains only discrete class labels
unique_classes = set(y_classified)
print(f"Unique classes in y_classified: {unique_classes}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_classified, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant', 'adaptive'],
}

# Initialize the MLP classifier
mlp_classifier = MLPClassifier(random_state=42, max_iter=1000)

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=mlp_classifier, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_mlp_classifier = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# Predict on the test set using the best model
y_pred = best_mlp_classifier.predict(X_test_scaled)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Calculate additional metrics
FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
TP = np.diag(conf_matrix)
TN = conf_matrix.sum() - (FP + FN + TP)
FPR = FP / (FP + TN)
specificity = TN / (TN + FP)
error_rate = 1 - accuracy

# Print all metrics
print(f"Acurácia: {accuracy:.2f}")
print(f"Taxa de erro: {error_rate:.2f}")
print(f"Precisão: {precision:.2f}")
print(f"Especificidade: {np.mean(specificity):.2f}")
print(f"Sensibilidade (Recall): {recall:.2f}")
print(f"FPR: {np.mean(FPR):.2f}")
print(f"F1 Score: {f1:.2f}")

# Plot the confusion matrix for the best model
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Distribuíveis', 'Classe 1 ou 2', 'Classe 3 ou 4', 'Turbidez Alta'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão para \'Turbidez\'')
plt.show()

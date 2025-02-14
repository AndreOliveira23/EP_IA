import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# Load the dataset using csv module
file_path = 'turb_stats_2020.csv'  # Update with the actual path
with open(file_path, newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=';')
    header = next(datareader)
    data = [row for row in datareader]

# Select relevant features and target variable
feature_indices = [header.index(col) for col in ['latitude', 'longitude', 'ano', 'nuturb', 'ambiente']]
target_index = header.index('medturb')

# Extract features and target data, replacing commas with periods for numeric conversion
X = [[float(row[i].replace(',', '.')) for i in feature_indices] for row in data]
y = [float(row[target_index].replace(',', '.')) for row in data]

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

y_classified = [classify_turbidity(value) for value in y]

# Verify that y_classified contains only discrete class labels
unique_classes = set(y_classified)
print(f"Unique classes in y_classified: {unique_classes}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_classified, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the KNN classifier and evaluate performance for different k values
for i in range(1, 26):
    knn_classifier = KNeighborsClassifier(n_neighbors=i)
    knn_classifier.fit(X_train_scaled, y_train)
    
    # Predict on the test set
    y_pred = knn_classifier.predict(X_test_scaled)
    
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
    print(f"k = {i}")
    print(f"Acurácia: {accuracy:.2f}")
    print(f"Taxa de erro: {error_rate:.2f}")
    print(f"Precisão: {precision:.2f}")
    print(f"Especificidade: {np.mean(specificity):.2f}")
    print(f"Sensibilidade (Recall): {recall:.2f}")
    print(f"FPR: {np.mean(FPR):.2f}")
    print(f"F1 Score: {f1:.2f}")

# Optionally plot the confusion matrix for the best k value
best_k = 9  # You can choose the best k value based on the metrics
knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
knn_classifier.fit(X_train_scaled, y_train)
y_pred = knn_classifier.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Distribuíveis','Classe 1 ou 2', 'Classe 3 ou 4', 'Turbidez Alta'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão para \'Turbidez\'')
plt.show()

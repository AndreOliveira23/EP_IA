import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset using csv module
file_path = 'od_stats_2020.csv'
with open(file_path, newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=';')
    header = next(datareader)
    data = [row for row in datareader]

# Select relevant features and target variable
feature_indices = [header.index(col) for col in ['latitude', 'longitude', 'ano', 'nuod', 'ambiente']]
target_index = header.index('medod')

# Extract features and target data, replacing commas with periods for numeric conversion
X = [[float(row[i].replace(',', '.')) for i in feature_indices] for row in data]
y = [float(row[target_index].replace(',', '.')) for row in data]

# Classify target variable based on the given ranges
def classify_medod(value):
    if value >= 6.0:
        return 1
    elif value >= 5.0:
        return 2
    elif value >= 4.0:
        return 3
    elif value >= 2.0:
        return 4
    else:
        return 5  # For values less than 2.0, which don't fall into any of the specified classes

y_classified = [classify_medod(value) for value in y]

# Convert data into latitude and longitude for plotting
latitude_index = header.index('latitude')
longitude_index = header.index('longitude')

latitudes = [float(row[latitude_index].replace(',', '.')) for row in data]
longitudes = [float(row[longitude_index].replace(',', '.')) for row in data]

# Define colors for each class
colors = ['green', 'blue', 'orange', 'red', 'black']
color_map = [colors[value - 1] for value in y_classified]

# Create a scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(longitudes, latitudes, c=color_map, alpha=0.6)

# Create a legend
legend_labels = ['Classe 1: ≥ 6.0 mg/L', 'Classe 2: ≥ 5.0 mg/L', 'Classe 3: ≥ 4.0 mg/L', 'Classe 4: ≥ 2.0 mg/L', 'Classe 5: < 2.0 mg/L']
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(len(legend_labels))]
plt.legend(handles, legend_labels, loc='upper right')

# Set plot title and labels
plt.title('Scatter Plot of Points Based on medod Classification')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the KNN model
knn = KNeighborsRegressor(n_neighbors=5)

# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')

# Calculate and print the mean RMSE from cross-validation
rmse_scores = (-cv_scores) ** 0.5
mean_rmse = rmse_scores.mean()

print(f"Mean RMSE from 5-Fold Cross-Validation: {mean_rmse:.2f}")

# Show the plot
plt.show()


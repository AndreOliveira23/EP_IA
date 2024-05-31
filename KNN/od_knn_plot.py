import csv
import matplotlib.pyplot as plt

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

# Classify target variable based on the condition
y_classified = [0 if value < 7 else 1 for value in y]

# Convert data into latitude and longitude for plotting
latitude_index = header.index('latitude')
longitude_index = header.index('longitude')

latitudes = [float(row[latitude_index].replace(',', '.')) for row in data]
longitudes = [float(row[longitude_index].replace(',', '.')) for row in data]

# Create a scatter plot
plt.figure(figsize=(10, 6))
colors = ['red' if value == 0 else 'blue' for value in y_classified]
plt.scatter(longitudes, latitudes, c=colors, label=['medod < 7', 'medod >= 7'], alpha=0.6)

# Set plot title and labels
plt.title('Scatter Plot of Points Based on medod Classification')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Create a legend
plt.legend(['medod < 7', 'medod >= 7'], loc='upper right')

# Show the plot
plt.show()

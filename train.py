
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Machine Learning model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Model accuracy:", round(accuracy * 100, 2), "%")

# Save the trained model
joblib.dump(model, "iris_model.pkl")

# Save the flower names
joblib.dump(iris.target_names, "flower_names.pkl")

print("Model and flower names saved successfully!")

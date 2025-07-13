# Build Your First Machine Learning Model
# ---------------------------------------

# Step 1: Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 2: Load the Iris dataset
# This is a built-in dataset containing measurements of different iris flowers
iris = load_iris()
X = iris.data  # Features (sepal length, petal length, etc.)
y = iris.target  # Labels (flower species: setosa, versicolor, virginica)

# Step 3: Split the dataset into training and testing sets
# This lets us train the model on one part of the data and test it on another
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train the model
# We use Logistic Regression, a simple and commonly used ML algorithm
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
predictions = model.predict(X_test)

# Step 6: Evaluate the model's performance
# We'll use accuracy, which tells us how many predictions were correct
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# Step 7 (Optional): Predict on new data
# Let's predict the species for a new flower based on its measurements
sample = [[5.1, 3.5, 1.4, 0.4]]
prediction = model.predict(sample)
print("Predicted class:", iris.target_names[prediction[0]])

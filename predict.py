import os 
os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np

# Load and preprocess MNIST test data
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1)

# Load model
model = keras.models.load_model("final_model.keras")

# Get predictions
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_classes == y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Compare predictions with actual labels for a few examples
num_examples = 10000
print("\nComparing predictions with actual labels:")
print("Index | Predicted | Actual")
print("-" * 25)
# Show test results that doesn't match
for i in range(num_examples):
    if predicted_classes[i] != y_test[i]:
        print(f"{i:5d} | {predicted_classes[i]:9d} | {y_test[i]}")

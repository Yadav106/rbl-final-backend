import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# 1. Load the trained model
model = load_model("model.h5")

# 2. Read the test data CSV
data = pd.read_csv("test_keypoint.csv", header=None)

# 3. Extract features (skip the label column)
X_test = data.iloc[:, 1:].values.astype(np.float32)

# 4. Predict probabilities
y_pred_probs = model.predict(X_test)

# 5. Get predicted class indices
y_pred = np.argmax(y_pred_probs, axis=1)

# 6. (Optional) Manually decode if you know class labels
class_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
decoded_labels = [class_names[i] for i in y_pred]

accuracy = 0

# 7. Print the results
for i, label in enumerate(decoded_labels):
    print(f"Sample {i+1}: Predicted label = {label}")
    print(f"Actual label = {data.iloc[i, 0]}")  # Assuming the first column is the actual label
    if label == data.iloc[i, 0]:
        accuracy += 1
    # save the result to a file
    with open("predictions.txt", "a") as f:
        f.write(f"Sample {i+1}: Predicted label = {label}, Actual label = {data.iloc[i, 0]}\n")

accuracy = accuracy / len(decoded_labels) * 100
print(f"Accuracy: {accuracy:.2f}%")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Credit Card Fraud Dataset
print("Loading Credit Card Dataset...")

data = pd.read_csv("creditcard.csv")

# Separate Features and Target
X = data.drop('Class', axis=1)   # Features
y = data['Class']                # Target (0 = Genuine, 1 = Fraud)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Build Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the Model
print("Training Credit Card Fraud Detection Model...")
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\nModel Evaluation Results")
print("------------------------")
print(f"Test Accuracy : {accuracy:.4f}")
print(f"Test Loss     : {loss:.4f}")

# Predict on a Sample Transaction
sample_transaction = X_test[:1]
prediction = model.predict(sample_transaction)

print("\nSample Transaction Prediction")
print("------------------------------")
print("Fraud Transaction" if prediction[0][0] > 0.5 else "Genuine Transaction")

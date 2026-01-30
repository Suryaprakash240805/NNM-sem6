import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import os

print("Checking for creditcard.csv...")
DATA_PATH = 'creditcard.csv'

if os.path.exists(DATA_PATH):
    print("Dataset found! Loading real data...")
    df = pd.read_csv(DATA_PATH)
else:
    print("Dataset NOT found. Generating synthetic data that mimics Kaggle Credit Card dataset (V1-V28, Time, Amount, Class)...")
    # Synthetic Kaggle-like data: 10,000 samples, 30 features
    n_samples = 10000
    features = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    X_synthetic = np.random.randn(n_samples, 30)
    # Class: 1 for fraud (0.5%), 0 for normal
    y_synthetic = np.random.choice([0, 1], size=n_samples, p=[0.995, 0.005])
    df = pd.DataFrame(X_synthetic, columns=features)
    df['Class'] = y_synthetic

# Separate Features and Labels
X = df.drop('Class', axis=1)
y = df['Class']

# Normalize Time and Amount (Common practice for this dataset)
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))

model = models.Sequential([
    # Input Layer: 30 neurons (one for each feature)
    # First Hidden Layer: 16 neurons
    layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2), # Prevent overfitting
    
    # Second Hidden Layer: 8 neurons
    layers.Dense(8, activation='relu'),
    
    # Output Layer: 1 neuron (Sigmoid for probability)
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

print("\nStarting Training...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    class_weight=class_weights,
    verbose=1
)

print("\nEvaluating on Test Set:")
results = model.evaluate(X_test, y_test)
print(f"Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
print(f"Precision: {results[2]:.4f}, Recall: {results[3]:.4f}")

# Final Prediction Example
sample_pred = model.predict(X_test[:5])
print("\nSample Probabilities (First 5 test cases):")
print(sample_pred.flatten())
print("Actual Labels:")
print(y_test[:5].values)

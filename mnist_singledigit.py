import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load MNIST Dataset
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to range [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the Model
print("Training MNIST model...")
model.fit(X_train, y_train, epochs=5, verbose=1)

# Select a Single Test Image
index = 0   # You can change this index
single_image = X_test[index]
actual_label = y_test[index]

# Preprocess Single Image
single_image_input = np.expand_dims(single_image, axis=0)

# Display the Digit Image
plt.imshow(single_image, cmap='gray')
plt.title("Test Digit Image")
plt.axis('off')
plt.show()

# Predict the Digit
test_image_input = np.expand_dims(single_image, axis=0)
prediction = model.predict(single_image_input)
predicted_digit = np.argmax(prediction)

# Display Results
print("\nSingle Digit Recognition Result")
print("--------------------------------")
print(f"Predicted Digit : {predicted_digit}")
print(f"Actual Digit    : {actual_label}")

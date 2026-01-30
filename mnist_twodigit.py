import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST Dataset
print("Loading MNIST dataset...")

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the Model
print("Training the model...")
model.fit(X_train, y_train, epochs=5, verbose=1)

# Select Two Test Images
index1 = 3
index2 = 8

image1 = X_test[index1]
image2 = X_test[index2]

label1 = y_test[index1]
label2 = y_test[index2]

# Display Both Digit Images
plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title("Digit 1")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image2, cmap='gray')
plt.title("Digit 2")
plt.axis('off')

plt.show()

# Predict Both Digits
images_input = np.array([image1, image2])
predictions = model.predict(images_input)

predicted_digit1 = np.argmax(predictions[0])
predicted_digit2 = np.argmax(predictions[1])

# Display Results
print("Two Digit Recognition Result")
print("-----------------------------")
print(f"Predicted Digits : {predicted_digit1}, {predicted_digit2}")
print(f"Actual Digits    : {label1}, {label2}")

import tensorflow as tf
import numpy as np
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(3, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='mean_squared_error',
              metrics=['accuracy'])
X = np.array([[0,0,0], [0,1,1], [1,0,1], [1,1,1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)
print("TensorFlow Training Started...")
history = model.fit(X, y, epochs=1000, verbose=0)
print("Training Completed.")
print("\nFinal Predictions:")
predictions = model.predict(X)
print(predictions)
print("\nActual Targets:")
print(y)
final_loss = history.history['loss'][-1]
print(f"\nFinal Loss: {final_loss:.4f}")

#!/usr/bin/env python3.13
import os
# Silence TensorFlow info/warning logs and oneDNN messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models

def run_inference_tf():
    # 1. Input layer values
    # free: 1, win: 0, offer: 1
    # Replacing np.array with tf.constant
    X = tf.constant([[1.0, 0.0, 1.0]], dtype=tf.float32)
    
    # 2. Weights and Biases (Using tf.constant instead of np.array)
    # Hidden Layer Weights (3 inputs, 2 neurons)
    W1 = tf.constant([
        [0.5, 0.4],
        [-0.2, 0.1],
        [0.3, -0.5]
    ], dtype=tf.float32)
    b1 = tf.zeros(2, dtype=tf.float32)
    
    # Output Layer Weights (2 hidden neurons, 1 output)
    W2 = tf.constant([
        [0.7],
        [0.2]
    ], dtype=tf.float32)
    b2 = tf.zeros(1, dtype=tf.float32)
    
    # 3. Build Model
    model = models.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(2, activation='relu', name='hidden_layer'),
        layers.Dense(1, activation='sigmoid', name='output_layer')
    ])
    
    # 4. Set weights manually
    hidden_layer = model.get_layer('hidden_layer')
    hidden_layer.set_weights([W1, b1])
    
    output_layer = model.get_layer('output_layer')
    output_layer.set_weights([W2, b2])
    
    # 5. Forward Pass
    prediction = model.predict(X, verbose=0)
    spam_probability = float(prediction[0][0])
    
    # --- Results ---
    print("--- Spam Classification (TensorFlow Only) ---")
    print(f"Input Features:             {X.numpy()[0]}")
    print(f"Final Spam Probability:     {spam_probability:.6f}")
    
    return spam_probability

if __name__ == "__main__":
    run_inference_tf()

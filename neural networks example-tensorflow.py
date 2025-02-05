import tensorflow as tf
import numpy as np

# Generate random data

num_samples = 2000
features = np.random.rand(num_samples, 3) * [2000, 3, 5]  # Random values for [sqft, bathrooms, bedrooms]
rent_prices = features @ [1.5, 500, 700] + np.random.randn(num_samples) * 100  # Add some noise

# Split into training and testing sets
train_features = features[:1700]
test_features = features[1700:]
train_labels = rent_prices[:1700]
test_labels = rent_prices[1700:]

# Normalize the features
mean = train_features.mean(axis=0)
std = train_features.std(axis=0)
train_features = (train_features - mean) / std
test_features = (test_features - mean) / std

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(3,)),  # First hidden layer with 128 neurons
    tf.keras.layers.Dropout(0.2),  # Dropout to prevent overfitting
    tf.keras.layers.Dense(64, activation='relu'),  # Second hidden layer with 64 neurons
    tf.keras.layers.Dropout(0.2),  # Dropout to prevent overfitting
    tf.keras.layers.Dense(1)  # Output layer for predicting rent prices
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Set up early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_features, train_labels,
    epochs=200, batch_size=32,
    validation_split=0.2, verbose=1,
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_mae = model.evaluate(test_features, test_labels, verbose=0)
print(f"Test Mean Absolute Error: {test_mae}")

# Make predictions
predictions = model.predict(test_features)

# Print predictions and actual values for the first 5 test samples
print("\n--- Predictions vs Actual Values (First 5 Test Samples) ---")
for i in range(20):
    print(f"Predicted Rent: {predictions[i][0]:.2f}, Actual Rent: {test_labels[i]:.2f}")

# Calculate and print a normalized error index
normalized_error = np.mean(np.abs((predictions.flatten() - test_labels) / test_labels))
print(f"\nNormalized Error Index (0 to 1): {normalized_error:.4f}")

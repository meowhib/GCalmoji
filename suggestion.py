import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Set parameters
input_dim = len(features)
output_dim = len(outfit_classes)
hidden_layers = 2
hidden_units = 128
dropout_rate = 0.2

# Define the model
model = Sequential()
model.add(LSTM(hidden_units, input_shape=(None, input_dim), return_sequences=True))
model.add(Dropout(dropout_rate))

for _ in range(hidden_layers - 1):
    model.add(LSTM(hidden_units, return_sequences=True))
    model.add(Dropout(dropout_rate))

model.add(Dense(output_dim, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_inputs, train_outputs, epochs=20, batch_size=32, validation_data=(val_inputs, val_outputs))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_inputs, test_outputs)
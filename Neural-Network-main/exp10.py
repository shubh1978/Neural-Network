import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, SimpleRNN
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

print("===== CNN + RNN ADVANCED EXPERIMENT =====")

# -------------------------------
# 1. DATA GENERATION (CNN)
# -------------------------------

samples = 100
img_size = 8

X_cnn = np.random.rand(samples, img_size, img_size, 1)
y_cnn = np.random.randint(0, 2, samples)

y_cnn_cat = to_categorical(y_cnn, 2)

# Split
split = int(0.8 * samples)
X_train_cnn, X_test_cnn = X_cnn[:split], X_cnn[split:]
y_train_cnn, y_test_cnn = y_cnn_cat[:split], y_cnn_cat[split:]

# -------------------------------
# 2. CNN MODEL
# -------------------------------

cnn_model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(8,8,1)),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), activation='relu'),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- CNN TRAINING ---")
history_cnn = cnn_model.fit(
    X_train_cnn, y_train_cnn,
    epochs=10,
    validation_data=(X_test_cnn, y_test_cnn),
    verbose=1
)

# -------------------------------
# 3. CNN EVALUATION
# -------------------------------

cnn_loss, cnn_acc = cnn_model.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print("\nCNN Accuracy:", cnn_acc)

pred_cnn = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
true_cnn = np.argmax(y_test_cnn, axis=1)

cm_cnn = confusion_matrix(true_cnn, pred_cnn)
print("CNN Confusion Matrix:\n", cm_cnn)

# -------------------------------
# 4. CNN GRAPH
# -------------------------------

plt.figure(figsize=(6,4))
plt.plot(history_cnn.history['accuracy'], label='Train Acc')
plt.plot(history_cnn.history['val_accuracy'], label='Val Acc')
plt.title("CNN Accuracy")
plt.legend()
plt.grid()
plt.show()

# -------------------------------
# 5. DATA GENERATION (RNN)
# -------------------------------

timesteps = 6
features = 1

X_rnn = np.random.rand(samples, timesteps, features)
y_rnn = np.random.randint(0, 2, samples)

y_rnn_cat = to_categorical(y_rnn, 2)

X_train_rnn, X_test_rnn = X_rnn[:split], X_rnn[split:]
y_train_rnn, y_test_rnn = y_rnn_cat[:split], y_rnn_cat[split:]

# -------------------------------
# 6. RNN MODEL
# -------------------------------

rnn_model = Sequential([
    SimpleRNN(32, activation='tanh', return_sequences=False, input_shape=(timesteps, features)),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- RNN TRAINING ---")
history_rnn = rnn_model.fit(
    X_train_rnn, y_train_rnn,
    epochs=10,
    validation_data=(X_test_rnn, y_test_rnn),
    verbose=1
)

# -------------------------------
# 7. RNN EVALUATION
# -------------------------------

rnn_loss, rnn_acc = rnn_model.evaluate(X_test_rnn, y_test_rnn, verbose=0)
print("\nRNN Accuracy:", rnn_acc)

pred_rnn = np.argmax(rnn_model.predict(X_test_rnn), axis=1)
true_rnn = np.argmax(y_test_rnn, axis=1)

cm_rnn = confusion_matrix(true_rnn, pred_rnn)
print("RNN Confusion Matrix:\n", cm_rnn)

# -------------------------------
# 8. RNN GRAPH
# -------------------------------

plt.figure(figsize=(6,4))
plt.plot(history_rnn.history['accuracy'], label='Train Acc')
plt.plot(history_rnn.history['val_accuracy'], label='Val Acc')
plt.title("RNN Accuracy")
plt.legend()
plt.grid()
plt.show()

# -------------------------------
# 9. TESTING CUSTOM INPUTS
# -------------------------------

sample_img = np.random.rand(1,8,8,1)
cnn_test_pred = cnn_model.predict(sample_img)
print("\nCNN Test Prediction:", np.argmax(cnn_test_pred))

sample_seq = np.random.rand(1, timesteps, features)
rnn_test_pred = rnn_model.predict(sample_seq)
print("RNN Test Prediction:", np.argmax(rnn_test_pred))

# -------------------------------
# 10. FINAL SUMMARY
# -------------------------------

print("\n===== FINAL RESULTS =====")
print("CNN Accuracy:", cnn_acc)
print("RNN Accuracy:", rnn_acc)

print("\n===== EXPERIMENT COMPLETED SUCCESSFULLY =====")
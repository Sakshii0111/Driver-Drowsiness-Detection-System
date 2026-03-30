# train_model.py
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# =========================
# 1. PARAMETERS
# =========================
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20

# =========================
# 2. LOAD DATASET
# Dataset folder structure:
# dataset/
#    train/
#       alert/
#       drowsy/
#    test/
#       alert/
#       drowsy/
# =========================
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)
print("Class Names:", train_data.class_names)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# =========================
# 3. DATA AUGMENTATION
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
])

# Apply augmentation + normalization
train_data = train_data.map(lambda x, y: (data_augmentation(x) / 255.0, y))
test_data = test_data.map(lambda x, y: (x / 255.0, y))

# Performance optimization
train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
test_data = test_data.prefetch(buffer_size=tf.data.AUTOTUNE)

# =========================
# 4. BUILD CNN MODEL
# =========================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.6),
    layers.Dense(1, activation='sigmoid')
])

# =========================
# 5. COMPILE MODEL
# =========================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# 6. EARLY STOPPING
# =========================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# =========================
# 7. TRAIN MODEL
# =========================
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# =========================
# 8. SAVE MODEL
# =========================
model.save("drowsiness_model.keras")
print("✅ Model Training Completed & Saved!")

# =========================
# 9. PLOT RESULTS
# =========================
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.show()
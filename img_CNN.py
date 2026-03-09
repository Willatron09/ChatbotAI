# A classical NN; adapted from https://www.tensorflow.org/tutorials/keras/classification/
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Loading the dataset
dataset = keras.utils.image_dataset_from_directory(
    "C:/Users/willi/Desktop/chatbotAI/Alcohol_Dataset",
    image_size=(128,128),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_dataset = keras.utils.image_dataset_from_directory(
    "C:/Users/willi/Desktop/chatbotAI/Alcohol_Dataset",
    image_size=(128,128),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=123
)
#output_classes = 3

print(dataset.class_names)

dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

## Build the model
model = keras.Sequential([
    keras.Input(shape=(128, 128, 3)),
    keras.layers.Rescaling(1./255),

    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),

    keras.layers.Conv2D(32, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(128, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

## Train the model

history = model.fit(
    dataset,
    validation_data=val_dataset,
    epochs=15,
    callbacks=[early_stop]
)

model.save("alcohol_classifier.h5")
## Evaluate the trained model
test_loss, test_acc = model.evaluate(val_dataset, verbose=2)
print("\nTest accuracy:", test_acc)

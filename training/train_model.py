import os
import json
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "Mini Dog Breed Data")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15   # 🔼 increased epochs

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

num_classes = train_gen.num_classes

# ✅ Load VGG19 base
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# ✅ Freeze most layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

# ✅ Unfreeze top layers (fine-tuning)
for layer in base_model.layers[-4:]:
    layer.trainable = True

# ✅ Custom classifier
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ✅ Lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ✅ Train
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# ✅ Save trained model
model.save(os.path.join(BASE_DIR, "dogbreed.h5"))
print("Model saved as dogbreed.h5")

# ✅ Save labels
class_indices = train_gen.class_indices
labels = {v: k for k, v in class_indices.items()}

with open(os.path.join(BASE_DIR, "labels.json"), "w") as f:
    json.dump(labels, f)

print("Saved labels.json")

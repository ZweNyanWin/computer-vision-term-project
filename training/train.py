"""Train the Thai wooden handicraft classifier with transfer learning.

MobileNetV2 pretrained on ImageNet, frozen backbone + new classification head,
then a short fine-tuning pass. Designed to run on Google Colab (free GPU) or
locally. Expects the dataset produced by scraper.py:

  data/
    frog/*.jpg
    elephant/*.jpg
    ...

Run:
  pip install tensorflow
  python training/train.py

Outputs:
  model/classifier.h5        - trained model
  model/class_names.txt
  training/history.png       - accuracy/loss curves for the report
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN = 10
EPOCHS_FINETUNE = 5
SEED = 22  # workshop day :)


def class_weights():
    """Balance training since classes have unequal image counts."""
    counts = {}
    for cls in sorted(os.listdir(DATASET_DIR)):
        d = os.path.join(DATASET_DIR, cls)
        if os.path.isdir(d):
            counts[cls] = len(os.listdir(d))
    total = sum(counts.values())
    n = len(counts)
    return {i: total / (n * c) for i, (cls, c) in enumerate(sorted(counts.items()))}


def load_datasets():
    train_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    )
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        class_names,
    )


def build_model(num_classes):
    # Week 2 material as data augmentation: flips, rotation, zoom
    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.15),
    ])
    base = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs), base


def plot_history(histories, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(acc, label="train"); ax1.plot(val_acc, label="val")
    ax1.set_title("Accuracy"); ax1.legend()
    ax2.plot(loss, label="train"); ax2.plot(val_loss, label="val")
    ax2.set_title("Loss"); ax2.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print("saved", path)


def main():
    train_ds, val_ds, class_names = load_datasets()
    print("classes:", class_names)

    model, base = build_model(len(class_names))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    weights = class_weights()
    print("class weights:", weights)
    print("phase 1: training the new head (backbone frozen)")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FROZEN, class_weight=weights)

    print("phase 2: fine-tuning the top of the backbone")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE, class_weight=weights)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "classifier.h5"))
    with open(os.path.join(MODEL_DIR, "class_names.txt"), "w") as f:
        f.write("\n".join(class_names) + "\n")
    print("saved model/classifier.h5 and model/class_names.txt")

    plot_history([h1, h2], os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.png"))

    # confusion matrix for the report
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true += labels.numpy().tolist()
        y_pred += np.argmax(preds, axis=1).tolist()
    cm = tf.math.confusion_matrix(y_true, y_pred).numpy()
    print("confusion matrix (rows = true, cols = predicted):")
    print("labels:", class_names)
    print(cm)


if __name__ == "__main__":
    main()

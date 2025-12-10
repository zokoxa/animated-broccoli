import tensorflow as tf
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def preprocess(image, label):
    image = tf.image.resize(image, (config["data"]["image_size"], config["data"]["image_size"]))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def augment(image, label):
    aug = config["augmentations"]
    if aug.get("random_flip", False):
        image = tf.image.random_flip_left_right(image)
    if aug.get("random_brightness", 0) > 0:
        image = tf.image.random_brightness(image, aug["random_brightness"])
    if aug.get("random_rotation", 0) > 0 and tf.random.uniform([]) < 0.25:
        image = tf.image.rot90(image)

    return image, label


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    config["data"]["data_dir"],
    image_size=(config["data"]["image_size"], config["data"]["image_size"]),
    batch_size=config["training"]["batch_size"],
    label_mode="categorical",
)

train_ds = (
    train_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(2048)
    .prefetch(tf.data.AUTOTUNE)
)

def build_resnet():
    if config["model"]["type"] == "resnet50":
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=(config["data"]["image_size"], config["data"]["image_size"], 3),
            pooling="avg"
        )
    else:
        raise ValueError(f"Unsupported model: {config['type']}")
    x = tf.keras.layers.Dense(config["data"]["num_classes"], activation="softmax")(base.output)
    return tf.keras.Model(inputs=base.input, outputs=x)

model = build_resnet()

optimizer = tf.keras.optimizers.Adam(config["optimizer"]["learning_rate"])
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{config['checkpoint']['dir']}/resnet_epoch{{epoch}}.weights.h5",
        save_weights_only=True
    ),
    tf.keras.callbacks.TensorBoard(log_dir="logs")
]

model.fit(
    train_ds,
    epochs=config["training"]["epochs"],
    callbacks=callbacks
)

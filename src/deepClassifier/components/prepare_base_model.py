import tensorflow as tf
from pathlib import Path
from deepClassifier.entity.config_entity import PrepareBaseModelConfig
from deepClassifier import logging


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
        )
        logging.info("Fetch the VGG16 model such that it will not conatin the fully connected layers")
        self.save_model(path=self.config.base_model_path, model=self.model)
        logging.info("Save the model to the  base model directory")

    @staticmethod
    def _prepare_full_model(
        model: tf.keras.Model, classes, freeze_all: bool, freeze_till, learning_rate
    ):
        logging.info("Freeze all the layer weights of the VGG16")
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False
        logging.info("Add the Flatten layer")
        flatten_in = tf.keras.layers.Flatten()(model.output)
        logging.info("Add the Dense layer with activation function softmax")
        prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(
            flatten_in
        )
        full_model = tf.keras.Model(inputs=model.input, outputs=prediction)
        logging.info("Compile the full model")
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        full_model.summary()
        return full_model

    def update_base_model(self):
        logging.info("Start updating the Base model")
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
        )
        logging.info("Save the modeified model")
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

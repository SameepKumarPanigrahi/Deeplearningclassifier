from deepClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path
from deepClassifier import logging

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def train_valid_generator(self):
        logging.info("Start Normailizing all the image and mention the validaion split which is 20 percent")
        datagenerator_kwargs = dict(rescale=1.0 / 255, validation_split=0.20)
        logging.info("Set the image size, batch size and intepolation")
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        logging.info("Set the valid_generator for giving the testing data")
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator
        logging.info("Set the train_generator for giving the training data")
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list):
        logging.info("Define the steps per epoch")
        self.steps_per_epoch = (
            self.train_generator.samples // self.train_generator.batch_size
        )
        logging.info("Define the validations steps")
        self.validation_steps = (
            self.valid_generator.samples // self.valid_generator.batch_size
        )

        logging.info("Start the model training")
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list,
        )

        logging.info("Save the traind model")
        self.save_model(path=self.config.trained_model_path, model=self.model)

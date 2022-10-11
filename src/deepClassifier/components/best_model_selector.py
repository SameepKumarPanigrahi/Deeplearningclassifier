import tensorflow as tf
import shutil
from pathlib import Path
from deepClassifier.config.configuration import BestModelSelectorConfig
from box import ConfigBox
from deepClassifier.utils.common import *
import os


class BestModelSelector:
    def __init__(self, config: BestModelSelectorConfig):
        self.config = config
        
    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split = 0.30
        ) 
        
        dataflow_kwargs = dict(
            target_size = self.config.params_image_size[:-1],
            batch_size = self.config.params_batch_size,
            interpolation = "bilinear"
        )
        valid_datagenarator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self.valid_generator = valid_datagenarator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
        
    @staticmethod
    def load_model(path:Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def evaluation(self):
        self.exisitng_model_score = None
        if(os.path.exists(self.config.existing_model_path)):
            self.model = self.load_model(self.config.existing_model_path)
            self._valid_generator()
            self.score = self.model.evaluate(self.valid_generator)
            scores = {"loss": self.score, "accuracy": self.score[1]}
            self.exisitng_model_score = ConfigBox(scores)
        self.new_model_score = load_json(self.config.score_file_path)
        
    def select_best_model(self):
        #newly_trained_model_directory = os.path.dirname(self.config.trained_model_path)
        final_destination = os.path.dirname(self.config.existing_model_path)
        if (self.exisitng_model_score == None):
            shutil.copy(self.config.trained_model_path, final_destination)
        else:
            if(self.new_model_score.accuracy > self.exisitng_model_score.accuracy):
                os.remove(self.config.existing_model_path)
                shutil.copy(self.config.trained_model_path, final_destination)
            
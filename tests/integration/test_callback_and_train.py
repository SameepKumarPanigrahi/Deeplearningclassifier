from deepClassifier.entity.config_entity import PrepareCallbacksConfig,TrainingConfig
from deepClassifier.components import PrepareCallBack, Training
from deepClassifier.utils.common import create_directories
import tensorflow as tf
import os

class Test_callback_And_Train:
    prepare_call_back_config = PrepareCallbacksConfig(
        root_dir="tests/data/prepare_callbacks",
        tensorboard_root_log_dir="tests/data/prepare_callbacks/tensorboard_log_dir",
        checkpoint_model_filepath="tests/data/prepare_callbacks/checkpoint_dir/model.h5"
    )
    training_config = TrainingConfig(
        root_dir="tests/data/training",
        trained_model_path="tests/data/training/model.h5",
        updated_base_model_path="tests/data/prepare_base_model/base_model_updated.h5",
        training_data="tests/data/PetImages",
        params_epochs=2,
        params_batch_size=1,
        params_is_augmentation=True,
        params_image_size=[224, 224, 3]
    )
    def test_callback(self):
        model_ckpt_dir = os.path.dirname(self.prepare_call_back_config.checkpoint_model_filepath)
        create_directories([model_ckpt_dir, self.prepare_call_back_config.tensorboard_root_log_dir])
        prepare_callback = PrepareCallBack(config=self.prepare_call_back_config)
        lst = prepare_callback.get_tb_ckpt_callbacks()
        assert isinstance(lst[0], tf.keras.callbacks.TensorBoard)
        assert isinstance(lst[1], tf.keras.callbacks.ModelCheckpoint)
    
    def test_train_getbase_model(self):
        training = Training(config=self.training_config)
        training.get_base_model()
        assert isinstance(training.model, tf.keras.Model)
        
    def test_train_valid_generator(self):
        training = Training(config=self.training_config)
        training.train_valid_generator()
        assert isinstance(training.valid_generator, tf.keras.preprocessing.image.DirectoryIterator)
        assert isinstance(training.train_generator, tf.keras.preprocessing.image.DirectoryIterator)
    
    def test_train(self):
        #model_ckpt_dir = os.path.dirname(self.prepare_call_back_config.checkpoint_model_filepath)
        # create_directories([model_ckpt_dir, self.prepare_call_back_config.tensorboard_root_log_dir])
        prepare_callback = PrepareCallBack(config=self.prepare_call_back_config)
        lst = prepare_callback.get_tb_ckpt_callbacks()
        training = Training(config=self.training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(callback_list=lst)
        ext = os.path.splitext(self.training_config.trained_model_path)[-1].lower()
        assert ext == '.h5'
        
    
    
    
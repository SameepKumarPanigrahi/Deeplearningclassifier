from deepClassifier.entity.config_entity import PrepareCallbacksConfig
from deepClassifier.components import PrepareCallBack
from deepClassifier.utils.common import create_directories
import tensorflow as tf
import os 


class Test_PrepareCallback:
    prepare_call_back_config = PrepareCallbacksConfig(
        root_dir="tests/data/prepare_callbacks",
        tensorboard_root_log_dir="tests/data/prepare_callbacks/tensorboard_log_dir",
        checkpoint_model_filepath="tests/data/prepare_callbacks/checkpoint_dir/model.h5"
    )
    def test_callback(self):
        model_ckpt_dir = os.path.dirname(self.prepare_call_back_config.checkpoint_model_filepath)
        create_directories([model_ckpt_dir, self.prepare_call_back_config.tensorboard_root_log_dir])
        prepare_callback = PrepareCallBack(config=self.prepare_call_back_config)
        lst = prepare_callback.get_tb_ckpt_callbacks()
        assert isinstance(lst[0], tf.keras.callbacks.TensorBoard)
        assert isinstance(lst[1], tf.keras.callbacks.ModelCheckpoint)
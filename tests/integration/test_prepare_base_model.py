from deepClassifier.entity import PrepareBaseModelConfig
from deepClassifier.components import PrepareBaseModel 
from pathlib import Path
import os 
import tensorflow as tf 

class Test_PrepareBaseModel:
    prepare_base_model_config = PrepareBaseModelConfig(
        root_dir="tests/data/prepare_base_model",
        base_model_path="tests/data/prepare_base_model/base_model.h5",
        updated_base_model_path="tests/data/prepare_base_model/base_model_updated.h5",
        params_image_size=[224, 224, 3],
        params_learning_rate=0.01,
        params_include_top=False,
        params_weights="imagenet",
        params_classes=2
    )
    
    def test_all_functionality(self):
        prepare_base_model = PrepareBaseModel(config=self.prepare_base_model_config)
        prepare_base_model.get_base_model()
        assert os.path.exists(self.prepare_base_model_config.base_model_path)
        ext = os.path.splitext(self.prepare_base_model_config.base_model_path)[-1].lower()
        assert ext == '.h5'
        
        prepare_base_model.update_base_model()
        assert os.path.exists(self.prepare_base_model_config.updated_base_model_path)
        ext = os.path.splitext(self.prepare_base_model_config.updated_base_model_path)[-1].lower()
        assert ext == '.h5'
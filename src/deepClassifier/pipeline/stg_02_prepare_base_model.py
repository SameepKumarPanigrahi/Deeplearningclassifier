from deepClassifier.config import ConfigurationManager
from deepClassifier import logger
from deepClassifier.components import PrepareBaseModel

STAGE_NAME = "Prepare base model"

def main():
    config = ConfigurationManager()
    prepare_base_model_config = config.get_prepare_base_model_config()
    prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
    prepare_base_model.get_base_model()
    prepare_base_model.update_base_model()

    
if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        main()
        logger.info(f">>>>> stage {STAGE_NAME} completed successfully <<<<< \n \n X{'=='*60}X")
    except Exception as e:
        logger.exception(e)
        raise e
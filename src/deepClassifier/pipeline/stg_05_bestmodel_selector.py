from deepClassifier.config import ConfigurationManager
from deepClassifier import logger
from deepClassifier.components.best_model_selector import BestModelSelector

STAGE_NAME = "Best Model Slection stage"


def main():
    config = ConfigurationManager()
    model_config = config.get_best_model_config()
    selector = BestModelSelector(model_config)
    selector.evaluation()
    selector.select_best_model()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        main()
        logger.info(
            f">>>>> stage {STAGE_NAME} completed successfully <<<<< \n \n X{'=='*60}X"
        )
    except Exception as e:
        logger.exception(e)
        raise e

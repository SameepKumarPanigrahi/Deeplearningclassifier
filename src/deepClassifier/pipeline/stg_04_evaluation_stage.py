from deepClassifier.config import ConfigurationManager
from deepClassifier import logger
from deepClassifier.components.evaluation import Evaluation

STAGE_NAME = "Evaluation stage"


def main():
    config = ConfigurationManager()
    validation_config = config.get_validation_config()
    evaluation = Evaluation(config=validation_config)
    evaluation.evaluation()
    evaluation.save_score()
    evaluation.log_into_mlflow()


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

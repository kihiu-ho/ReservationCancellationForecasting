from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
from steps.utils.ConfigLoader import ConfigLoader
from steps.preprocess_step import PreprocessStep
from steps.train_step import TrainStep
from steps.condition_step import ConditionStep
from steps.feature_engineering_step import FeatureEngineeringStep
from steps.utils.data_classes import PreprocessingData, FeaturesEngineeringData
from steps.config import (
    TRAINING_DATA_PATH,
    TrainerConfig,
    PreprocessConfig,
    FeatureEngineeringConfig,
)



import logging


from steps.condition_step import ConditionStep
from steps.config import ConditionConfig
from steps.utils.ConfigLoader import ConfigLoader

LOGGER = logging.getLogger(__name__)


def load_configuration(ti):
    config_loader = ConfigLoader()
    try:
        config_loader.load_config("/opt/airflow/config/config.yaml")
    except FileNotFoundError:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(current_dir), "config", "config.yaml")
        config_loader.load_config(config_path)

    if config_loader.config is None:
        raise ValueError("Unable to load the config properly!")

    ti.xcom_push(key='config', value=config_loader.config)

def condition_task_callable(**kwargs):

    config_loader = ConfigLoader("/opt/airflow/config/config.yaml")

    mlflow_run_info = kwargs['ti'].xcom_pull(task_ids='training', key='return_value')
    mlflow_run_id = mlflow_run_info.get("mlflow_run_id") if isinstance(mlflow_run_info, dict) else mlflow_run_info

    if not mlflow_run_id:
        raise ValueError("The required MLflow run_id is not provided or invalid")

    condition_checker = ConditionStep(criteria=0.05, metric="roc_auc", config_loader=config_loader)
    condition_checker(mlflow_run_id)


default_args = {
    "owner": "user",
    "depends_on_past": False,
    "retries": 0,
    "catchup": False,
}

with DAG(
        "training-pipeline",
        default_args=default_args,
        start_date=datetime(2023, 12, 19),
        tags=["training"],
        schedule=None,
) as dag:
    load_config_task = PythonOperator(
        task_id="load_configuration",
        python_callable=load_configuration,
    )

    preprocessing_task = PythonOperator(
        task_id="preprocessing",
        python_callable=PreprocessStep(inference_mode=False, preprocessing_data=PreprocessingData(
            train_path=PreprocessConfig.train_path,
            test_path=PreprocessConfig.test_path)),
        op_kwargs={"data_path": TRAINING_DATA_PATH},
    )

    feature_engineering_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=FeatureEngineeringStep(inference_mode=False, feature_engineering_data=FeaturesEngineeringData(
            train_path=FeatureEngineeringConfig.train_path,
            test_path=FeatureEngineeringConfig.test_path,
            encoders_path=FeatureEngineeringConfig.encoders_path)),
        op_kwargs={
            "train_path": PreprocessConfig.train_path,
            "test_path": PreprocessConfig.test_path,
        },
    )

    training_task = PythonOperator(
        task_id="training",
        python_callable=TrainStep(params=TrainerConfig.params),
        op_kwargs={
            "train_path": FeatureEngineeringConfig.train_path,
            "test_path": FeatureEngineeringConfig.test_path,
            "target": FeatureEngineeringConfig.target
        },
    )

    validation_task = PythonOperator(
        task_id="validation_task",
        python_callable=condition_task_callable,
        provide_context=True
    )

    # DAG flow
    load_config_task >> preprocessing_task >> feature_engineering_task >> training_task >> validation_task

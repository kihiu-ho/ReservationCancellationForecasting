from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
from steps.utils.ConfigLoader import ConfigLoader
from steps.preprocess_step import PreprocessStep
from steps.inference_step import InferenceStep
from steps.feature_engineering_step import FeatureEngineeringStep
from steps.utils.data_classes import PreprocessingData, FeaturesEngineeringData
from steps.config import (
    FeatureEngineeringConfig,
    INFERENCE_DATA_PATH,
    PreprocessConfig,
)



def load_configuration(ti):
    config_loader = ConfigLoader()
    try:
        config_loader.load_config("/opt/airflow/config/config.yaml")
    except FileNotFoundError:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        secondary_config_path = os.path.join(
            os.path.dirname(current_dir), "config", "config.yaml"
        )
        config_loader.load_config(secondary_config_path)

    if not config_loader.config:
        raise ValueError("Loaded configuration is empty. Verify the YAML file and the paths provided.")

    ti.xcom_push(key='config', value=config_loader.config)



default_args = {
    "owner": "user",
    "depends_on_past": False,
    "retries": 0,
    "catchup": False,
}

with DAG(
        "inference-pipeline",
        default_args=default_args,
        start_date=datetime(2023, 12, 20),
        tags=["inference"],
        schedule=None,
) as dag:
    # Task to load configuration
    load_config_task = PythonOperator(
        task_id="load_configuration",
        python_callable=load_configuration,
    )

    # Preparation (outside airflow task still allowed for simple configs)
    inference_mode = True
    preprocessing_data = PreprocessingData(
        batch_path=PreprocessConfig.batch_path
    )
    features_engineering_data = FeaturesEngineeringData(
        batch_path=FeatureEngineeringConfig.batch_path,
        encoders_path=FeatureEngineeringConfig.encoders_path,
    )

    preprocess_step = PreprocessStep(
        inference_mode=inference_mode,
        preprocessing_data=preprocessing_data
    )

    feature_engineering_step = FeatureEngineeringStep(
        inference_mode=inference_mode,
        feature_engineering_data=features_engineering_data
    )


    # define inference task callable to retrieve configuration from xcom:
    from steps.utils.ConfigLoader import ConfigLoader


    def inference_task_callable(batch_path, ti):
        config_dict = ti.xcom_pull(key='config', task_ids='load_configuration')

        if not config_dict:  # Verify presence of configuration; safeguard mechanism.
            raise ValueError(
                "Configuration data not found in XCom. Ensure config is properly loaded in the previous task!")

        config_loader = ConfigLoader()
        config_loader.config = config_dict

        # Explicit load call here, if ConfigLoader demands it explicitly
        if not hasattr(config_loader, 'config') or not config_loader.config:
            config_loader.load_config()

        # Instantiate inference step once config confirmed
        inference_obj = InferenceStep(config_loader=config_loader)

        # Now properly call inference_obj with loaded configurations
        response = inference_obj(batch_path=batch_path)

        return response


    # Define Airflow PythonOperators
    preprocessing_task = PythonOperator(
        task_id="preprocessing",
        python_callable=preprocess_step,
        op_kwargs={
            "data_path": INFERENCE_DATA_PATH
        }
    )

    feature_engineering_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering_step,
        op_kwargs={
            "batch_path": preprocessing_data.batch_path
        }
    )

    inference_task = PythonOperator(
        task_id="inference",
        python_callable=inference_task_callable,
        op_kwargs={
            "batch_path": features_engineering_data.batch_path
        }
    )

    # DAG task dependencies
    load_config_task >> preprocessing_task >> feature_engineering_task >> inference_task

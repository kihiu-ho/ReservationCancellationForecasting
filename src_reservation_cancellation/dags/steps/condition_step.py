import logging
from typing import Literal
import mlflow

LOGGER = logging.getLogger(__name__)


class ConditionStep:
    """Condition to register the model.

       Args:
           criteria (float): Coefficient applied to the metric of the model registered in the model registry.
           metric (str): Metric as a reference. Can be `["precision", "recall", "roc_auc"]`.
           config_loader (ConfigLoader): Instance of ConfigLoader.
       """

    def __init__(
            self,
            criteria: float,
            metric: Literal["roc_auc", "precision", "recall"],
            config_loader,
    ) -> None:
        self.criteria = criteria
        self.metric = metric
        self.mlflow_config = config_loader.get_mlflow_config()

        # Validate that essential keys exist in mlflow_config
        required_keys = ["tracking_uri", "artifact_location", "registered_model_name"]
        for key in required_keys:
            if key not in self.mlflow_config:
                raise ValueError(f"Missing required key '{key}' in MLflow config.")

    def __call__(self, mlflow_run_id: str) -> None:
        """
           Compare the metric from the last run to the model in the registry.
           If `metric_run > registered_metric*(1 + self.criteria)`, then the model is registered.
           """
        LOGGER.info(f"Run_id: {mlflow_run_id}")
        mlflow.set_tracking_uri(self.mlflow_config["tracking_uri"])

        # Get the metrics of the current run
        run = mlflow.get_run(run_id=mlflow_run_id)
        metric = run.data.metrics.get(self.metric)

        if metric is None:
            raise ValueError(
                f"Metric '{self.metric}' not found in the current run data."
            )

        # Search for registered models with the specified name
        registered_models = mlflow.search_registered_models(
            filter_string=f"name = '{self.mlflow_config['registered_model_name']}'"
        )

        if not registered_models:
            # No models registered yet, register the current one
            LOGGER.info("No registered models found. Registering a new model...")
            mlflow.register_model(
                model_uri=f"runs:/{mlflow_run_id}/{self.mlflow_config['artifact_location']}",
                name=self.mlflow_config["registered_model_name"],
            )
            LOGGER.info("New model registered.")
            return

        # Access the latest registered model
        latest_registered_model = registered_models[0]
        latest_version = latest_registered_model.latest_versions[0]

        if not latest_version:
            raise ValueError(
                "Unable to retrieve the latest version of the registered model."
            )

        # Get the metrics of the latest registered model
        registered_model_run = mlflow.get_run(latest_version.run_id)
        registered_metric = registered_model_run.data.metrics.get(self.metric)

        if registered_metric is None:
            raise ValueError(
                f"Metric '{self.metric}' not found in the registered model data."
            )

        # Compare metrics and register a new version if the current run's metric is better
        if metric > registered_metric * (1 + self.criteria):
            LOGGER.info(
                "Current model metric exceeds threshold. Registering a new version..."
            )
            mlflow.register_model(
                model_uri=f"runs:/{mlflow_run_id}/{self.mlflow_config['artifact_location']}",
                name=self.mlflow_config["registered_model_name"],
            )
            LOGGER.info("Model registered as a new version.")
        else:
            LOGGER.info(
                "Current model does not exceed the performance criteria. No new model version registered."
            )

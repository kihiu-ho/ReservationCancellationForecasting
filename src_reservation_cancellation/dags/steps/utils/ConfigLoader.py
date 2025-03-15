import os
import json
import yaml
class ConfigLoader:
    """
    Handles loading configurations from various sources.
    """

    def __init__(self, config_file=None):
        self.config = {}
        if config_file:
            self.load_config(config_file)

    def load_config(self, config_file: str):
        """
        Loads configuration from a JSON or YAML file.
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        file_ext = os.path.splitext(config_file)[-1].lower()
        if file_ext == ".yaml" or file_ext == ".yml":
            import yaml
            with open(config_file, "r") as f:
                self.config = yaml.safe_load(f)
        elif file_ext == ".json":
            with open(config_file, "r") as f:
                self.config = json.load(f)
        else:
            raise ValueError("Unsupported config file format. Use JSON or YAML.")

    def get_mlflow_config(self):
        """
        Get the MLflow configuration.
        Returns an empty dictionary if no "mlflow" section is found.
        """
        if not self.config:
            raise ValueError("Configuration not loaded. Call `load_config()` first.")
        return self.config.get("mlflow", {})


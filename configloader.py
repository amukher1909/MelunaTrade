import yaml
import logging
import logging.config
from typing import Any, Dict

class ConfigLoader:
    """
    A utility class to load, manage, and provide access to the system's
    configuration settings from a YAML file.
    """
    def __init__(self, config_path: str):
        """
        Initializes the ConfigLoader with the path to the configuration file.

        Args:
            config_path (str): The file path to the main config.yaml.
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Loads the YAML configuration file.

        Returns:
            Dict[str, Any]: A dictionary containing the configuration settings.
        
        Raises:
            FileNotFoundError: If the configuration file cannot be found.
            yaml.YAMLError: If the configuration file is malformed.
        """
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at '{self.config_path}'")
            raise
        except yaml.YAMLError as e:
            print(f"Error: Malformed YAML in configuration file: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value for a given key.

        Args:
            key (str): The configuration key to retrieve.
            default (Any, optional): A default value to return if the key is not found.
                                     Defaults to None.

        Returns:
            Any: The value of the configuration setting.
        """
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """
        Allows dictionary-style access to configuration settings.
        e.g., config_loader['data_settings']
        """
        return self.config[key]
    
def setup_logging(config: Dict[str, Any]) -> None:
    """
    Sets up the logging configuration for the entire application.

    This function uses the configuration details from the provided dictionary
    to set up handlers, formatters, and log levels. It's designed to
    handle failure gracefully by defaulting to a basic configuration if
    the provided one is invalid, aligning with our fail-safe mentality.

    Args:
        config (Dict[str, Any]): A dictionary with the logging configuration,
                                 typically loaded from the YAML file.
    """
    try:
        logging.config.dictConfig(config)
        logging.info("Logging configured successfully from config file.")
    except (ValueError, TypeError, AttributeError) as e:
        # Fallback to a basic configuration if the one in the file is malformed
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s")
        logging.warning(f"Could not configure logging from dict: {e}. Using basic config.")

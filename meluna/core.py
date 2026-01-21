import yaml
import logging
import logging.config
from typing import Any, Dict
import os
import re
from pathlib import Path

class ConfigLoader:
    """Loads and manages system configuration from YAML file."""
    def __init__(self, config_path: str):
        """Initialize with path to config file."""
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
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
        """Get configuration value by key with optional default."""
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to config."""
        return self.config[key]
    
    def get_all(self) -> Dict[str, Any]:
        """Return entire configuration dictionary."""
        return self.config
    
def setup_logging(config: Dict[str, Any]) -> None:
    """Configure application logging with fallback to basic config."""
    try:
        logging.config.dictConfig(config)
        logging.info("Logging configured successfully from config file.")
    except (ValueError, TypeError, AttributeError) as e:
        # Fallback to a basic configuration if the one in the file is malformed
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s")
        logging.warning(f"Could not configure logging from dict: {e}. Using basic config.")

def get_next_version_path(base_results_dir: Path, backtest_name: str) -> Path:
    """Generate next version path for backtest results."""
    backtest_path = base_results_dir / backtest_name
    backtest_path.mkdir(exist_ok=True)
    
    # Find all existing version directories (e.g., 'v1', 'v2', etc.)
    existing_versions = [
        d for d in os.listdir(backtest_path)
        if os.path.isdir(backtest_path / d) and re.match(r'^v(\d+)$', d)
    ]
    
    if not existing_versions:
        next_version_num = 1
    else:
        # Extract the numbers, find the max, and add 1
        max_version = max(int(re.search(r'(\d+)', v).group()) for v in existing_versions)
        next_version_num = max_version + 1
        
    next_version_path = backtest_path / f"v{next_version_num}"
    next_version_path.mkdir(exist_ok=True)
    
    return next_version_path
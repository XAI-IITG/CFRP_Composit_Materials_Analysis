import yaml
import os
# from src.utils.logger import get_logger # Assuming logger is setup

# logger = get_logger(__name__)
# DEFAULT_CONFIG_PATH = 'src/config/config.yaml' # Default path relative to project root if run from root

# _config = None # Global cache for the loaded config

def load_config(config_path=None):
    """
    Loads the YAML configuration file.
    Uses a cached version after the first load.
    Searches for config.yaml in standard locations if path is not provided.
    """
    # global _config
    # if _config is not None and config_path is None: # Use cached if no specific path given
    #     return _config

    # search_paths = []
    # if config_path:
    #     search_paths.append(config_path)
    # search_paths.extend([
    #     DEFAULT_CONFIG_PATH,
    #     'config/config.yaml', # If run from src/ or other subdirs
    #     os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml') # Relative to this file
    # ])

    # loaded_path = None
    # for path_option in search_paths:
    #     if os.path.exists(path_option):
    #         loaded_path = path_option
    #         break
    
    # if not loaded_path:
    #     logger.error(f"Configuration file not found in searched paths: {search_paths}")
    #     raise FileNotFoundError(f"Configuration file not found. Searched: {search_paths}")

    # try:
    #     with open(loaded_path, 'r') as f:
    #         current_config = yaml.safe_load(f)
    #     if _config is None and config_path is None: # Cache it if it's the first default load
    #         _config = current_config
    #     logger.info(f"Configuration loaded successfully from: {loaded_path}")
    #     return current_config
    # except yaml.YAMLError as e:
    #     logger.error(f"Error parsing YAML configuration from {loaded_path}: {e}")
    #     raise
    # except Exception as e:
    #     logger.error(f"An unexpected error occurred while loading config from {loaded_path}: {e}")
    #     raise
    pass

if __name__ == '__main__':
    # try:
    #     config = load_config()
    #     if config:
    #         print("Configuration loaded:")
    #         # print(config)
    #         # Example: Accessing a config value
    #         # raw_data_path = config.get('paths', {}).get('raw_data', 'default/path/not/found')
    #         # print(f"Raw data path from config: {raw_data_path}")
    # except FileNotFoundError:
    #     print("config.yaml not found. Please ensure it exists in src/config/ or provide a path.")
    # except Exception as e:
    #     print(f"Error loading config: {e}")
    print("Configuration loader placeholder")



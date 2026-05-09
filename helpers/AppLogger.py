import logging
from .GetEnv import GetEnv
from pathlib import Path

class AppLogger():
    global _env
    _env = GetEnv.get_env_variables()
    def __init__(self, **kwargs):
        self.log_config = None
        self.log_filename = f"{_env['LOG_DIR']}/{kwargs['file_name']}.log"
        self.logging_level = None
        self.init_log_config = self.init_config()


    def init_config(self):

        Path(f"{_env['LOG_DIR']}").mkdir(exist_ok=True)

        logger = logging.getLogger(__name__)
        logging.basicConfig(filename=self.log_filename, level=logging.DEBUG, force=True)
        return logger
import logging

class AppLogger():
    def __init__(self):
        self.log_config = None
        self.log_filename = './logs/app.log'
        self.logging_level = None
        self.init_log_config = self.init_config()


    def init_config(self):
        logger = logging.getLogger(__name__)
        logging.basicConfig(filename=self.log_filename, level=logging.DEBUG, force=True)
        return logger
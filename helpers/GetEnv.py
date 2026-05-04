from dotenv import load_dotenv, dotenv_values
import os

class GetEnv:
    @staticmethod
    def get_env_variables():
        # Get absolute path of project root
        BASE_DIR = os.path.dirname('./')

        # Construct .env path
        env_path = os.path.join(BASE_DIR, ".env")


        # load_dotenv(env_path)     

        return dotenv_values(env_path)
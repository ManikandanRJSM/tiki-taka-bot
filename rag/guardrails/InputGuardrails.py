import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

class InputGuardrails:
    
    @staticmethod
    def machine_learning_guardrails(**kwargs):
        text = kwargs['user_input'].lower()
        _env = kwargs['_env']
        logger_object = kwargs['logger_object']
        lg_model = joblib.load(f"{_env['GUARDRAIL_MODEL_PATH']}/lg_ip_guardrails.pkl")
        vectorizer = joblib.load(f"{_env['GUARDRAIL_MODEL_PATH']}/vectorizer.pkl")
        logger_object.info(f"Loaded guardrail model {_env['GUARDRAIL_MODEL_PATH']}/lg_ip_guardrails.pkl")

        
        text_vector = vectorizer.transform([text])

        result = lg_model.predict(text_vector)

        logger_object.info(f"User query : {kwargs['user_input']}")
        logger_object.info(f"Model result : {result} - {'safe' if result[0] == 0 else 'un_safe'}")
        
        result = 'safe' if result[0] == 0 else 'un_safe'

        return result

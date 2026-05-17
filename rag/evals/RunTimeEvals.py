from ragas.metrics import DiscreteMetric

class RunTimeEvals():

    @staticmethod
    def ragas_eval(**kwargs):

        kwargs['logger_object'].info(f"Enters RAGAS Evals...........")
        correctness_metric = DiscreteMetric(
        name="correctness",
        prompt=f"""Compare the model response to the expected answer and determine if it's correct.

        Consider the response correct if it:
        1. Contains the key information from the expected answer
        2. Is factually accurate based on the provided context
        3. Adequately addresses the question asked

        Return 'pass' if the response is correct, 'fail' if it's incorrect.

        Question: {kwargs['user_question']}
        Model Response: {kwargs['llm_response']}

        Evaluation:""",
            allowed_values=["pass", "fail"],
        )

        kwargs['logger_object'].info(f"RAGAS results {correctness_metric}")
        kwargs['logger_object'].info(f"RAGAS results values {correctness_metric.value}")

        return correctness_metric
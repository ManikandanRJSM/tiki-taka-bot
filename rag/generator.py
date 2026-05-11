from helpers.GetEnv import GetEnv
from .module_helpers.GetConnection import GetConnection
import torch
from transformers import pipeline
import os
from helpers.AppLogger import AppLogger
from rank_bm25 import BM25Okapi
import re
from .evals.RunTimeEvals import RunTimeEvals


global logger

logger_object = AppLogger(file_name = 'generator_log')

logger = logger_object.init_log_config


def SearchSemantic(user_prompt, mode = 'CLI'):

    _env = GetEnv.get_env_variables()
    os.environ["HF_TOKEN"] = _env['HF_TOKEN']

    # Get and init the connection for collection and vector embedding model
    _con = GetConnection()
    client, embedding_model, reranker, collection = _con.initConnection()

    query_vector = embedding_model.encode(user_prompt) # Convert the user promt into tokens and vectors

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        where={'type' : 'match_result'}, # Meta data filter
        n_results=3
    )

    if len(results['documents'][0]) <= 0:
        return Generator(['No records available in the database'], user_prompt, mode, _env['LLM_MODEL_DEVICE'])
    bm25_results = init_bm(document = results, user_prompt = user_prompt, THRESHOLD = '0.5')

    if bm25_results['is_correct_document']:

        # Re-ranker machnism to strengthen the RAG from top n documents from bestmatching
        pairs = [[user_prompt, doc] for doc in bm25_results['top_n']]
        scores = reranker.predict(pairs)
        re_rank_doc = bm25_results['top_n'][scores.argmax()]
        
        return Generator(re_rank_doc, user_prompt, mode, _env['LLM_MODEL_DEVICE'])

    return "I'm sorry, I don't have enough information in my current knowledge base to answer this"


def Generator(results, user_question, query_from, device_type='cpu'):

    context = "\n".join(results)

    pipe = pipeline(
        "text-generation", 
        model=_env['LLM_MODEL_NAME'], 
        dtype=torch.bfloat16, 
        device_map=device_type
    )

    logger.info(f"Enters into LLM model used : {_env['LLM_MODEL_NAME']}, device_type - {device_type}")

    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {
            "role": "system",
            "content": "FIFA analyst. Answer only from context. Be concise."
        },
        {
            "role": "user", 
            "content": f"Context:\n{context}\n\nQuestion: {user_question}"
        },
    ]

    generate_kwargs = {
        "do_sample": True,
        "temperature": 0.7,
        "max_new_tokens": 1000,
        "top_k": 50,
        "top_p": 0.95
    }
    

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) #tokenize=False returns the strings instead of vectors
    outputs = pipe(prompt, **generate_kwargs)
    full_text = outputs[0]["generated_text"]

    logger.info(f'Token out - {len(pipe.tokenizer.encode(full_text))}')

    response = full_text[len(prompt):].strip()

    # Ragas Eval:
    ragas_results = RunTimeEvals.ragas_eval(llm_response = response, logger_object = logger, llm_prompt = messages, user_question = user_question)

    if ragas_results.value == 'pass':
        if query_from == 'CLI':
            return response
        else:
            return {
                'status' : 200,
                'status_message' : 'Sucess',
                'response' : response
            }
        
    return "I'm sorry, I don't have enough information in my current knowledge base to answer this"
    

def init_bm(**kwargs):

    logger.info(f"Entered into BM function - BM Threshold set to {kwargs['THRESHOLD']}")

    patterns_arr = ['Match: ', 'Date: ', 'Competition: ', 'Result: ', 'Venue city: ', 'Venue country: ', 'Is neutral: ', 'Winner: ']

    pattern = "|".join(map(re.escape, patterns_arr))

    results = kwargs['document']

    tokenized_corpus = [
        [item.strip() for item in re.sub(pattern, "", doc).lower().split('\n')]
        for doc in results['documents'][0]
    ]
    
    bm25 = BM25Okapi(tokenized_corpus)
    
    
    tokenized_query = kwargs['user_prompt'].lower().split()

    scores = bm25.get_scores(tokenized_query)
    top_n = bm25.get_top_n(tokenized_query, results['documents'][0], n=2)

    logger.info(f'BM Scores : {scores} - Top results : {top_n}')

    has_threshold_list = [score > float(kwargs['THRESHOLD']) for score in scores]

    relevane_document = [top_n[i] for i, val in enumerate(has_threshold_list) if val  ]

    if True in has_threshold_list: #score greater than 0.5 the retrival text has the relevancy
        return {
            'eval_message' : 'strong_match',
            'confidence': 'high',
            'is_correct_document': True,
            'scores': scores,
            'top_n': relevane_document
        }
    
    return {
            'eval_message' : 'weak_match',
            'confidence': 'low',
            'is_correct_document': False,
            'scores': scores,
            'top_n': top_n
        }
    

    # print("Scores:", scores)
    # print("Top results:", top_n)

def greet_user():

    print("Hi! I'm Tiki-Taka AI Agent 🤖⚽")
    print("Before we get started...")
    
    name = input("May I know your name? → ").strip()
    
    print(f"\nNice to meet you, {name}! 👋")
    print("How can I help you today?")

    logger.info('Starting session............')
    while True:
        user_question = input(f"\n{name}: ").strip()
        
        if not user_question:
            continue
        
        if user_question.lower() in ["exit", "quit", "bye"]:
            print(f"\nGoodbye {name}! See you soon! 👋⚽")
            break
        
        # Your ChromaDB query goes here
        print(f"Searching for: {user_question}...")
        print(SearchSemantic(user_question))

if __name__ == "__main__":

    logger.info('Enters Cli mode............')

    _env = GetEnv.get_env_variables()
    greet_user()
    logger.info('Session Ended............')

    
    
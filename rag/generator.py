from helpers.GetEnv import GetEnv
from .module_helpers.GetConnection import GetConnection
import torch
from transformers import pipeline
import os
from helpers.AppLogger import AppLogger
from rank_bm25 import BM25Okapi
import re


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

    score, top_n = init_bm(document = results, user_prompt = user_prompt, THRESHOLD = 0.5)
    print(score, top_n)
    exit()
    
    
   
    # pairs = [[user_prompt, doc] for doc in results['documents'][0]]

    # scores = reranker.predict(pairs)

    # doc = results['documents'][0][scores.argmax()]
    
    
    return Generator(results, user_prompt, mode, _env['LLM_MODEL_DEVICE'])


def Generator(results, user_question, query_from, device_type='cpu'):

    context = "\n".join(results['documents'][0])

    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    # model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")


    # response = ollama.chat(
    #     model="phi3",             # better than TinyLlama now!
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": "FIFA analyst. Answer only from context. Be concise."
    #         },
    #         {
    #             "role": "user",
    #             "content": f"Context:\n{context}\n\nQuestion: {user_prompt}"
    #         }
    #     ]
    # )

    # return response['message']['content']


    pipe = pipeline(
        "text-generation", 
        model=_env['LLM_MODEL_NAME'], 
        dtype=torch.bfloat16, 
        device_map=device_type
    )

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

    if query_from == 'CLI':
        return response
    else:
        return {
            'status' : 200,
            'status_message' : 'Sucess',
            'response' : response
        }
    

def init_bm(**kwargs):

    logger.info(f'Entered into BM function - BM Threshold set to {kwargs['THRESHOLD']}............')

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
    top_n = bm25.get_top_n(tokenized_query, results['documents'][0], n=3)

    logger.info(f'BM Scores : {scores} - Top results : {top_n}............')

    if scores < kwargs['THRESHOLD']:
        return {
            ''
        }

    return scores, top_n

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

    
    
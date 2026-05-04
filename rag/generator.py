from helpers.GetEnv import GetEnv
from .module_helpers.GetConnection import GetConnection
import torch
from transformers import pipeline
import os


def SearchSemantic(user_prompt):

    _env = GetEnv.get_env_variables()
    os.environ["HF_TOKEN"] = _env['HF_TOKEN']

    # Get and init the connection for collection and vector embedding model
    _con = GetConnection()
    client, embedding_model, collection = _con.initConnection()

    query_vector = embedding_model.encode(user_prompt) # Convert the user promt into tokens and vectors

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        where={'type' : 'match_result'}, # Meta data filter
        n_results=2
    )
    return Generator(results, user_prompt, _env['LLM_MODEL_DEVICE'])


def Generator(results, user_question, device_type='cpu'):

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
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    full_text = outputs[0]["generated_text"]
    response = full_text[len(prompt):].strip()

    return response


def greet_user():
    print("Hi! I'm Tiki-Taka AI Agent 🤖⚽")
    print("Before we get started...")
    
    name = input("May I know your name? → ").strip()
    
    print(f"\nNice to meet you, {name}! 👋")
    print("How can I help you today?")
    
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

    _env = GetEnv.get_env_variables()
    greet_user()

    
    
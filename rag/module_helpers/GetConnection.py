from helpers.GetEnv import GetEnv
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

class GetConnection:
    def __init__(self):
        self.embedding_model = None
        self.hf_token = None
        self.device = None
        self.chroma_collection = None

    def initConnection(self):
        _env = GetEnv.get_env_variables()

        self.embedding_model = _env['EMBEDDING_MODEL_NAME']
        self.hf_token = _env['HF_TOKEN']
        self.device = _env['EMBEDDING_MODEL_DEVICE']
        self.chroma_collection = _env['COLLECTION_NAME']

        # client = chromadb.Client()
        client = chromadb.PersistentClient(path=f"{_env['DATA_LAKE_PATH']}/vectors")

        embedding_model = SentenceTransformer( # Generates dimensions
            self.embedding_model,
            device=self.device
        )

        hf_ef = HuggingFaceEmbeddingFunction(
            api_key=self.hf_token,
            model_name=self.embedding_model
        )

        reranker = CrossEncoder(_env['RERANKER_MODEL_NAME'])

        collection = client.get_or_create_collection(name=self.chroma_collection, embedding_function=hf_ef)
        return client, embedding_model, reranker, collection
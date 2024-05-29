from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from chatglm_llm import ChatGLM
import sentence_transformers
import torch

# Global Parameters
EMBEDDING_MODEL = "bge-large-zh" #"text2vec"
VECTOR_SEARCH_TOP_K = 6
LLM_MODEL = "chatglm3-6b"
LLM_HISTORY_LEN = 3
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Show reply with source text from input document
REPLY_WITH_SOURCE = True

datapath = ""
embedding_model_dict = {
    "bge-large-zh": "F:\Agent\project\Langchain-Chatchat/bge-large-zh",
}

llm_model_dict = {

    "chatglm3-6b": "F:\Agent\project\Langchain-Chatchat\chatglm3-6b",
}


def init_cfg(LLM_MODEL, EMBEDDING_MODEL, LLM_HISTORY_LEN, V_SEARCH_TOP_K=6):
    global chatglm, embeddings, VECTOR_SEARCH_TOP_K
    VECTOR_SEARCH_TOP_K = V_SEARCH_TOP_K

    chatglm = ChatGLM()
    chatglm.load_model(model_name_or_path=llm_model_dict[LLM_MODEL])
    chatglm.history_len = LLM_HISTORY_LEN

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[EMBEDDING_MODEL],)
    embeddings.client = sentence_transformers.SentenceTransformer(embeddings.model_name,
                                                                  device=DEVICE)
    
    return embeddings


def get_db():
    embeddings = init_cfg(LLM_MODEL, EMBEDDING_MODEL, LLM_HISTORY_LEN)
    vector_store =FAISS.load_local('dataloader/',embeddings)
    return vector_store
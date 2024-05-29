from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from chatglm_llm import ChatGLM
import sentence_transformers
import torch
import os



# Global Parameters
EMBEDDING_MODEL = "bge-large-zh" #"text2vec"
VECTOR_SEARCH_TOP_K = 6
LLM_MODEL = "chatglm3-6b"
LLM_HISTORY_LEN = 3
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Show reply with source text from input document
REPLY_WITH_SOURCE = True


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


def init_knowledge_vector_store(filepath:str):
    if not os.path.exists(filepath):
        print("路径不存在")
        return None
    elif os.path.isfile(filepath):
        file = os.path.split(filepath)[-1]
        try:
            loader = UnstructuredFileLoader(filepath, mode="elements")
            docs = loader.load()
            print(f"{file} 已成功加载")
        except:
            print(f"{file} 未能成功加载")
            return None
    elif os.path.isdir(filepath):
        docs = []
        for file in os.listdir(filepath):
            fullfilepath = os.path.join(filepath, file)
            try:
                loader = UnstructuredFileLoader(fullfilepath, mode="elements")
                docs += loader.load()
                print(f"{file} 已成功加载")
            except:
                print(f"{file} 未能成功加载")

    vector_store = FAISS.from_documents(docs, embeddings)

    vector_store.save_local('test_vector_db')
    print('faiss saved!')
    
    return vector_store




if __name__ == "__main__":
    init_cfg(LLM_MODEL, EMBEDDING_MODEL, LLM_HISTORY_LEN)
    vector_store = None
    while not vector_store:
        filepath = input("Input your local knowledge file path 请输入本地知识文件路径：")
        vector_store = init_knowledge_vector_store(filepath)



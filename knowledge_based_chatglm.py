from langchain.chains import RetrievalQA
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from chatglm_llm import ChatGLM

import torch
import re
from get_prompt import get_prompt_template

from config import init_cfg, get_db, VECTOR_SEARCH_TOP_K

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

# 加载faiss向量库，用于知识召回
# vector_store = get_db()

global chatglm, embeddings
chatglm = ChatGLM()
chatglm.load_model(model_name_or_path=llm_model_dict[LLM_MODEL])


json_pattern = "请以json格式抽取"  # 匹配特定问题类型 
fund_pattern = "统计图"
# prompt_template = ["basic_info_qa", "core_indicator_statistics", "structured_info_extraction", "fund_analysis_report"]
def get_knowledge_based_answer(query, vector_store, chat_history=[]):
    global chatglm, embeddings

    is_structured = re.search(json_pattern, query)
    id_fund = re.search(fund_pattern, query)
    if is_structured:
        prompt_template = "structured_info_extraction"
    elif id_fund:
        prompt_template = "fund_analysis_report"
    else:
        prompt_template = "basic_info_qa"
    
    # print("prompt_template is :", prompt_template)
    system_template = get_prompt_template(prompt_template,  "default")
    # print(system_template)

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chatglm.history = chat_history
    knowledge_chain = RetrievalQA.from_llm(
        llm=chatglm,
        retriever=vector_store.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K}),
        prompt=prompt
    )

    knowledge_chain.return_source_documents = True

    result = knowledge_chain({"query": query})
    chatglm.history[-1][0] = query
    return result, chatglm.history



if __name__ == "__main__":
    embeddings = init_cfg(LLM_MODEL, EMBEDDING_MODEL, LLM_HISTORY_LEN)
    vector_store =FAISS.load_local('dataloader/',embeddings)

    while True:
        query = input("Input your question 请输入问题：")
        resp, history = get_knowledge_based_answer(query=query,
                                                   vector_store=vector_store)

        print(resp["result"])

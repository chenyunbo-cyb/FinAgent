import json

from knowledge_based_chatglm import get_knowledge_based_answer
from config import init_cfg
from config import LLM_MODEL, EMBEDDING_MODEL, LLM_HISTORY_LEN
from langchain_community.vectorstores import FAISS

# Show reply with source text from input document
REPLY_WITH_SOURCE = False

embeddings = init_cfg(LLM_MODEL, EMBEDDING_MODEL, LLM_HISTORY_LEN)
# 加载faiss向量库，用于知识召回
vector_store =FAISS.load_local('dataloader/',embeddings)


# companies = ["财通裕惠","东方精选" "东方新能源汽车主题", "方正富邦科技" , "方正富邦天璇", "红塔红土盛世", "华泰柏瑞成长智选", "华泰柏瑞消费成长", "南方宝丰"\
#              "南方远见", "平安鼎信", "浦银安盛", "人保利丰", "新华聚利" ,"医疗产业ETF", "招商瑞丰"]

# answers = ["这是问题0的答案", "这是问题1的答案"]
questions = []

with open("question.json", "r", encoding='utf-8') as f:
     for line in f:
        data = json.loads(line)
        # print(data)
        questions.append(data)


# prompt_template = ["basic_info_qa", "core_indicator_statistics", "structured_info_extraction", "fund_analysis_report"]

# for i in range(len(questions)):
with open('answer.json', 'w', encoding='utf-8') as f:
    for i in range(len(questions)):
        print(questions[i]['question'])
        query = questions[i]['question']
        # print(query)
        resp, history =  get_knowledge_based_answer(query=query, vector_store=vector_store)
    
        print("--------------------- answer: ---------------------------")
        if REPLY_WITH_SOURCE:
            print(resp)
        else:
            print(resp["result"])
        print("---------------------------------------------------------")
        # print(resp["result"])
        questions[i]['answer'] = resp["result"]
        # 对每一个问题，一旦得到答案，就立即将它写入 json 文件
        json.dump(questions[i], f, ensure_ascii=False)
        f.write("\n")


# # 将更新后的问题列表写入到一个新的json文件中
# with open('answer.json', 'w', encoding='utf-8') as f:
#     for question in questions:
#         json.dump(question, f, ensure_ascii=False)
#         f.write("\n")
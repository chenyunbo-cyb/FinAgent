
1、模型下载
1.1下载大模型文件：
https://huggingface.co/THUDM/chatglm3-6b/tree/main

将其所有文件下载，放置在./chatglm3-6b 文件夹下

1.2、下载Embedding文件：
https://huggingface.co/BAAI/bge-large-zh/tree/main

将其所有文件下载，放置在./bge-large-zh 文件夹下

2、环境配置
创建一个虚拟环境，并在虚拟环境内安装项目的依赖
pip install -r requirements.txt 

运行 init_knowledge_base.py ： 构建知识向量库
运行 knowledge_based_chatglm.py ：进行问答
运行 answer.py ： 根据question.json进行回答并生成answer.json

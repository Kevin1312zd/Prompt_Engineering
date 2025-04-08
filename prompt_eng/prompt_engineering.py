import os
import sys
import time
from typing import List, Any

import json5
from openai import base_url

sys.path.append("result/")
sys.path.append("..")
sys.path.append("../")

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI,OpenAIEmbeddings,OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 设置OpenAI API密钥
os.environ["SILICON_API_KEY"] = "sk-oqcowacpwgdulsodmslioidrxxdptzbenrbsvgductzxtxlu"

from file_util import logger_config

class prompt_generator:
    def __init__(self,file_json_path,prompt_number):
        self.qa_chain = None
        self.remote_model_list = None
        self.result_list=[]
        self.question_param_list=None
        self.rag_file_path_list=None

        self.file_json_path=file_json_path    #输入json文档
        self.read_json_file()
        self.prompt_number=prompt_number      #生成提示词数量
        self.rag_creator()                    #初始化生成rag
        
    def read_json_file(self):
        # 初始化模型参数，json5字符串
        with open(self.file_json_path, encoding="utf-8") as f1:
            params = json5.load(f1)
            self.remote_model_list = params["remote_model_list"]
            self.question_param_list = params["question_param_list"]
            self.rag_file_path_list = params["rag_file_path_list"]
        print("0000",self.question_param_list)


    #生成rag
    def rag_creator(self):
        # 加载文档
        doc=[]
        for rag_file_path in self.rag_file_path_list:
            print("prompt_rag_kb_txt\\"+rag_file_path)
            loader = TextLoader("prompt_rag_kb_txt\\"+rag_file_path, encoding="utf-8")
            #documents = loader.load()
            #doc = loader.load()[0].page_content
            #doc = loader.load()
            doc.extend(loader.load())

        # 分割文本块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000, #按需修改
            chunk_overlap=200
        )
        #chunks = text_splitter.split_text(doc)
        chunks = text_splitter.split_documents(doc)

        # 3. 创建向量数据库
        api_keys = os.getenv("SILICON_API_KEY")
        # embeddings = OpenAIEmbeddings()
        # vectorstore = Chroma.from_documents(
        #     documents=chunks,
        #     embedding=embeddings,
        #     persist_directory="./db"  # 向量数据库存储路径
        # )

        # client = OpenAI(api_key = api_keys,base_url="https://api.siliconflow.cn/v1")
        # responses = client.embeddings.create(
        #     input='Silicon flow embedding online: fast, affordable, and high-quality embedding services. come try it out!',
        #     model='BAAI/bge-m3'
        # )

        embeddings = OpenAIEmbeddings(
            model="BAAI/bge-m3",
            openai_api_key = api_keys,
            openai_api_base = "https://api.siliconflow.cn/v1"
        )

        vectorstore = Chroma.from_documents(
             documents=chunks,
             #embedding=responses.data[0].embedding,
             embedding=embeddings,
             persist_directory="./db"  # 向量数据库存储路径
         )


        # 4. 创建问答链
        remote_model_param_dict = self.remote_model_list[1]#这个通过qt界面改成可选项
        # 核心模型设置 ✅
        llm = ChatOpenAI(
            api_key=remote_model_param_dict["api_key"],
            base_url=remote_model_param_dict["base_url"],
            model=remote_model_param_dict["model"],
            temperature=remote_model_param_dict["temperature"],
            # max_tokens=1024,
        )

        prompt_template = """You are building a prompt to address user requirement.Based on the given reference prompt 
        {question}
        , Please select only one optimal prompt engineering strategy from my prompt knowledge base 
        {context}
        ,please reconstruct and optimize it using the optimal strategy. You can add, modify, or delete prompts. During the optimization, you can incorporate any thinking models.
        Please output strictly according to the following format without any additional words:
        
        Prompt:<Optimal prompt words>
        Strategy：<Only one correspondent optimal prompt engineering strategy>"""

        _prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context","question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": _prompt},
            return_source_documents=True
        )

    #输入提示词，先通过json文件输入
    def prompt_generate(self):
        for i in range(self.prompt_number):
            for question in self.question_param_list:
                result = self.qa_chain.invoke({"query": question})
                self.result_list.append(result["result"])

class prompt_evaluator:
    def __init__(self,result_list,file_json_path):
        self.qa_chain = None
        self.result_list = result_list
        self.file_json_path=file_json_path
        self.evaluator_model=self.prompt_evaluator_model()
        self.flag=0

    #读入评估大模型
    def prompt_evaluator_model(self):
        with open(self.file_json_path, encoding="utf-8") as f1:
            params = json5.load(f1)
            remote_model_list = params["remote_model_list"]
        return remote_model_list[1]

    #评估提示词
    def evalute_prompt(self):
        # 核心模型设置 ✅
        llm = ChatOpenAI(
            api_key=self.evaluator_model["api_key"],
            base_url=self.evaluator_model["base_url"],
            model=self.evaluator_model["model"],
            temperature=self.evaluator_model["temperature"],
            # max_tokens=1024,
        )

        prompt_template = """Evaluate the two prompts inputed, according to the following iutput format,
        prompt1:{prompt1}
        and
        prompt2:{prompt2},
        determine which one better.
        Provide your analysis and the choice you believe is bette, just strictly output the better prompt according to the following output format without any additional words.
        
        Input Format:(if input content includes analysis(Analysis:<>),just ignore)
        Prompt:<Optimal Prompt Words>
        Strategy：<Correspondent optimal prompt engineering strategy>
        
        Output Format:
        Prompt:<Optimal Prompt Words>
        Strategy：<Correspondent optimal prompt engineering strategy>
        Analysis:<（使用中文输出）分析最优提示词的优势>
                
        """
        # prompt_template = """Evaluate the two responses,
        # prompt1:{prompt1}
        # and
        # prompt2:{prompt2},
        # determine which one better.
        # Provide your analysis and the choice you believe is better, using XML tags to encapsulate your response.
        # <analyse>Some analysis</analyse>
        # <choose>prompt1/prompt2(the better prompt in your opinion)</choose>
        # """

        _prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["prompt1","prompt2"]
        )

        best_prompt = self.result_list[0]

        for i in range(len(self.result_list)-1):
            # 修正字典定义
            input_dict = {"prompt1": best_prompt,"prompt2":self.result_list[i + 1]}
            #input_dict = {"prompt1": RunnablePassthrough(), "prompt2":RunnablePassthrough()}
            chain = (
                    _prompt
                    | llm
            )

            best_prompt = chain.invoke(input_dict).content
            self.flag =self.flag + 1
        print("faslfdajsdf", best_prompt)
        return best_prompt



if __name__ == "__main__":
    # 初始化日志格式
    date_str = time.strftime('%Y-%m-%d', time.localtime())
    time_str = time.strftime('%H_%M_%S', time.localtime())
    result_path = os.path.join(os.getcwd(), f"result/{date_str}")
    logger = logger_config(result_path, f"{time_str}.txt")

    logger.info(f"-------------------------start new experiment-------------------------------")
    a = prompt_generator("prompt_enginnering.json",2)
    a.prompt_generate()
    logger.info("")
    logger.info(f"用户输入的原始prompt0是【 {a.question_param_list[0]} 】\n")
    logger.info(f"optimizer基于prompt0生成的多个提示词包括:\n")
    logger.info("——————————————————————————————————————————————————————————————————————————————————————————————————")
    for i in a.result_list:
        logger.info(f"【 {i} \n】\n")
        logger.info("——————————————————————————————————————————————————————————————————————————————————————————————————")
    b = prompt_evaluator(a.result_list,"prompt_enginnering.json")
    best_prompt = b.evalute_prompt()
    logger.info(f"经过evaluator评估得到的最优提示词是【 {best_prompt} 】\n")

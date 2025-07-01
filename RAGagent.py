import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
# import torch

# model = SentenceTransformer('./BAAI/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a').to('cuda:0')

# sentences = ['word is bad', 'life is beautiful']

# embbeddings = model.encode(sentences)
# print(type(embbeddings))

class retriever:
    """
    # 目前的功能
    retriver移除了稀疏检索,只包含稠密检索,
    如果后续仍有需要可以添加BM25等相关稀疏检索技术
    所以检索的方式目前只包含了内积Inner Product的形式
    
    # 工作原理
    工作原理是使用sentence transformer将日志文件(json: List[Dict[Any,Any]])解析,
    所有json文件的文本内容将会另外存储于docstore.json文件(或另外指令,后同)中,
    所有的embedding向量将储存在 vector_db.npy(numpy数组,后续可以考虑更高效的处理方式)
    """
    def __init__(self, config:dict):
        self.device = config['device']
        self.reconstruct = config['reconstruct']
        self.model_name = config['model_name']
        self.model = SentenceTransformer(self.model_name).to(self.device)
        self.index_mode = config['index_mode']
        self.rawdata_dir = config['rawdata_dir']
        if config.get('vector_db_path', None):
            self.vector_db_path = config['vector_db_path']
        else:
            self.vector_db_path = './vector_db.npy'
        if config.get('docstore_path', None):
            self.docstore_path = config['docstore_path']
        else:
            self.docstore_path = './docstore.json'
        
        if self.reconstruct or not os.path.exists(self.vector_db_path) or not os.path.exists(self.docstore_path):
            print("----------将要重构向量数据库和文本切片-----------\n")
            print("--------------这可能需要花费几分钟--------------\n")
            file_paths = []
            docs = []
            sentences = []
            for dirpath, dirnames, filenames in os.walk(self.rawdata_dir):
                for filename in filenames:
                    full_path = os.path.join(dirpath, filename)
                    file_paths.append(full_path)
            for file_path in file_paths:
                with open(file_path, 'r') as f:
                    content = json.load(f)

                if not isinstance(content, list):
                    raise ValueError("必须是list[dict]的形式！")
                else:
                    for tmp in content:
                        if not isinstance(tmp, dict):
                            raise ValueError("必须是list[dict]的形式！")           
                f.close()
                docs.extend(content)
                for log in content:
                    log_inf = str(log)[1:-1].replace('\n', ' ')
                    sentences.append(log_inf)
            embeddings = self.model.encode(sentences)
            with open(self.docstore_path, 'w+') as f:
                json.dump(docs, f)
            f.close()
            np.save(self.vector_db_path, embeddings)
            print("------------------数据处理完成-------------------\n")
            norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
            self.embeddings = embeddings
            self.docs = docs
        else:
            self.embeddings = np.load(self.vector_db_path)
            with open(self.docstore_path, 'r') as f:
                self.docs = json.load(f)
            f.close()
        
        if self.index_mode == 'default':
            n, d = self.embeddings.shape
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.embeddings)
        else:
            raise ValueError("未知模式")
        print("--------------------构建完成---------------------\n")
    
    def get_retrieval(self, query, k=10, level=0, proctol_name = None):
        '''
        :param k, 最近邻向量数量
        :param level, 结果过滤等级
        level = 0: 不做处理
        level = 1: 根据日志文件报头严格过滤一次
        '''
        # if isinstance(querys, str):
        #     # 说明只有一个查询
        #     querys = [querys]
        q_emb = self.model.encode([query])
        D, I = self.index.search(q_emb, k)
        index_list = I[0]
        retri_docs = [self.docs[t] for t in index_list]
        if proctol_name != None:
            # exact match
            for doc in self.docs:
                if doc.get('Name', None) == proctol_name:
                    retri_docs.append(doc)
        if level > 0:
            tmp = []
            for doc in retri_docs:
                name = doc.get('Name', None)
                if name.lower() in query.lower():
                    tmp.append(doc)
            retri_docs = tmp

        return retri_docs

if __name__ == '__main__':
    pass



        



        
from RAGagent import retriever
import time

config = {
        'device' : 'cuda', 
        
        'reconstruct' : False, # 如果需要重建或是第一次构建索引则设置为true 否则为false
        
        'model_name' : "./BAAI/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", # 指定sentence transformer(BERT模型)
        
        'index_mode' : "default", # 索引模式目前只支持default,为最简单的内积
        
        'rawdata_dir' : 'logdata', # 原始数据文件夹

        'vector_db_path': None, # 向量库存储位置

        'docstore_path': None, # 处理后文本存储位置
    }

R = retriever(config=config)

query = 'what is OSPF/3/VLINK_NBR_CHG_DOWN ?'

t0 = time.time()
for i in range(1000):
    r = R.get_retrieval(query, k=10, level=1)
    if i == 0:
        print(r)
t1 = time.time()
print(f"检索1000次,每次平均用时:{(t1-t0)}ms")
    
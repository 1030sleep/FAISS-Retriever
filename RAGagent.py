import faiss
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('./BAAI/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a').to('cuda:0')

sentences = ['word is bad', 'life is beautiful']

embbeddings = model.encode(sentences)
print(type(embbeddings))
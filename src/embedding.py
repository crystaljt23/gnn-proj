from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, models
import torch
import numpy as np
import pandas

titles = pandas.read_csv("table.csv")["title"].tolist()
model = SentenceTransformer('all-MiniLM-L6-v2')
train_embeddings = model.encode(titles)

# pca decomp to reduce dimensionality to a manageable size
# from https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/distillation/dimensionality_reduction.py
new_dimension = 32
pca = PCA(n_components=new_dimension)
pca.fit(train_embeddings)
pca_comp = np.asarray(pca.components_)

dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=new_dimension, bias=False, activation_function=torch.nn.Identity())
dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
model.add_module('dense', dense)

title_embeddings = model.encode(titles)

print(title_embeddings[:10])

with open('embeddings.npy', 'wb') as f:
    np.save(f, title_embeddings)
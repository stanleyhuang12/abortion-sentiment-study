import pandas as pd
import numpy as np
import tensorflow as tf 
import json 
from pathlib import Path
import os
import glob
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt 


file_path = Path(os.getcwd())

list_of_datasets = glob.glob(str(file_path/"data/*"))

for dataset_path in list_of_datasets: 
    with open(dataset_path) as f: 
        df_dict = json.load(f)
        df_1 = pd.DataFrame(df_dict)
        
    
with open(file_path/"data") as f: 
    df_dict = f.load()
    df_1 = pd.DataFrame(df_dict)
    
df_1[df_1['author_flair_text'].isnull() == False] 

## Step 1 convert news sources -> political affiliation 

## Step 2: topical feature 
## Step 3: score
## Step 4: number of comments 
## Step 5: Word count

df_1['article'].iloc[3]
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_vectors = embedding_model.encode(df_1['article'].iloc[3])
df_1['text_embedding'] = df_1['article'].apply(lambda x: embedding_model.encode(x))
df_1.to_csv('dataset.csv', index=False)


text_embed_matrix = np.vstack(df_1['text_embedding'])
pca_model = PCA(n_components=2)
pca_model.fit(text_embed_matrix)
pca_model.components_
pca_model.explained_variance_
cumulative_var = np.cumsum(pca_model.explained_variance_ratio_)

plt.plot(cumulative_var)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()


two_d_plot = pca_model.fit_transform(text_embed_matrix)
df_two_components = pd.DataFrame(two_d_plot)


df_1['domain'].value_counts().head(60)

df_two_components['domain'] = df_1['domain']

#1 is democrat, left-leaning
#2 is republican, right-leaning
#3 is independent, centrist
#4 is mixed 
#5 is not detailed 

domain_category_map = {
    'theguardian.com': 1,
    'bbc.com': 4,
    'cnbc.com': 4,
    'myspace.com': 5,
    'bbc.co.uk': 4,
    'apnews.com': 4,
    'nbcnews.com': 1,
    'cbsnews.com': 1,
    'usatoday.com': 4,
    'foxnews.com': 2,
    'reuters.com': 4,
    'cnn.com': 1,
    '178.62.16.199': 5,
    'yahoo.com': 4,
    'punjabkesari.in': 5,
    'dailymail.co.uk': 2,
    'abcnews.go.com': 1,
    'businessinsider.com': 4,
    'independent.co.uk': 1,
    'slideshare.net': 5,
    'nypost.com': 2,
    "thebiafrastar.com": 2,
    "npr.org": 1,
    "huffingtonpost.com": 1, 
    "thedailybeast.com": 2,
    "dailywire.com": 2,
    "politico.com": 3,
    "medium.com": 4,
    "wired.com": 1,
    "globalnews.ca": 3,
    "buzzfeednews.com": 1,
    "google.com": 4,
    "www-m.cnn.com": 1,
    "theatlantic.com": 1,
    "amp.cnn.com": 1,
    "forbes.com": 3,
    "cbc.ca": 1
}

index_to_color = { 1: "blue", 2: "red", 3:"green", 4: "purple", 5: "grey"}

df_two_components['pol_affil'] = df_two_components['domain'].apply(lambda x: domain_category_map.get(x, 5))
df_two_components['color'] = df_two_components['pol_affil'].map(index_to_color)
plt.scatter(df_two_components[0], df_two_components[1], color=df_two_components['color'])
plt.xlabel('First dimension')
plt.ylabel('Second dimension')
plt.show()

len(df_1['text_embedding'].loc[0])
df_1['WordCount'].loc[0]


tsne = TSNE(n_components=3, verbose=1)
tsne_results = tsne.fit_transform(text_embed_matrix)
tsne_df_res = pd.DataFrame(tsne_results)
tsne_df_res['color'] = df_two_components['color']

tsne_df_res.to_csv("tsne_df_res.csv", index=False)
fig = plt.figure()
ax = plt.axes(projection='3d')
scatter = ax.scatter3D(tsne_df_res[0], tsne_df_res[1], tsne_df_res[2], c=tsne_df_res['color'], cmap='viridis')
ax.set_xlabel('First dimension')
ax.set_ylabel('Second dimension')
ax.set_zlabel('Third dimension')
plt.show()


# Pro abortiion
# Pro life 
# Remove stop words
# Comment analysis 



# linear model that captures count
# A measure of how well something is doing 
# Linear regression
# Try cutting the first 100 characters 
# Pull soemthing interpretable out of it 
## Reduce topic -> do abortion or other group by similar topic
## Run linear inference
## Linear discriminant analysis (two dimension analysis to split pro choice and pro )
#feature extraction 

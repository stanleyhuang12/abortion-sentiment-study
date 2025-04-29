import pandas as pd
import numpy as np
import json 
from pathlib import Path
import os
import glob
from sentence_transformers import SentenceTransformer
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from tools import * 
import matplotlib.pyplot as plt 
    
## Import files 



file_path = Path(os.getcwd())
list_of_datasets = glob.glob(str(file_path/"data/*"))

df_dictionary = {}

for dataset_path in list_of_datasets: 
    with open(dataset_path) as f: 
        df_dict = json.load(f)
        df = pd.DataFrame(df_dict)
        key = dataset_path.split('/')[-1].split('.')[0]
        df_dictionary[key] = df
    
for key, df in df_dictionary.items(): 
    key_name = str(key)
    df.to_csv(key_name+'_file.csv', index=False)
    
prochoice_df = pd.read_csv('subreddit_prochoice_file.csv')
prolife_df = pd.read_csv('subreddit_prolife_file.csv') 


# drop_indices = np.random.choice(prolife_df.index, size=1100, replace=False)
# prolife_df = prolife_df.drop(index=drop_indices)
# prolife_df = prolife_df.drop(prolife_df['WordCount'].idxmax())

## Embed the documents using sentence transformers 
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
prochoice_df['text_embedding'] = prochoice_df['article'].apply(lambda x: embedding_model.encode(x))
prolife_df['text_embedding'] = prolife_df['article'].apply(lambda x: embedding_model.encode(x))

prochoice_df.reset_index(drop=True, inplace=True)
prolife_df.reset_index(drop=True, inplace=True)

## Upload into pickle file 
prochoice_df.to_pickle('subreddit_prochoice_embed.pkl')
prolife_df.to_pickle('subreddit_prolife_embed.pkl')

## Import as new dataframe for further analysis 

prochoice_df = pd.read_pickle('subreddit_prochoice_embed.pkl')
prolife_df = pd.read_pickle('subreddit_prolife_embed.pkl')

combined_df_transformer_1 = prochoice_df._append(prolife_df)

prochoice_embed_matrix = np.vstack(prochoice_df['text_embedding'])
prolife_embed_matrix = np.vstack(prolife_df['text_embedding'])
combined_embed_matrix = np.vstack((prochoice_embed_matrix, prolife_embed_matrix))

cat_label = np.zeros(prochoice_embed_matrix.shape[0]).reshape(-1, 1)
cat_label = np.vstack((cat_label, np.ones(prolife_embed_matrix.shape[0]).reshape(-1, 1)))

### Compute PCA projections and cumulative explained variance 

df_prolife_proj, prolife_explained_variance = compute_proj_and_variance(prolife_embed_matrix, "PCA", n_components=10)
df_prochoice_proj, prochoice_explained_variance = compute_proj_and_variance(prochoice_embed_matrix, "PCA", n_components=10)
df_proj, cum_explained_variance = compute_proj_and_variance(combined_embed_matrix, "PCA", n_components=10)
df_proj['category'] = cat_label.flatten()

### Visualize projected PCA  and t-SNE 

visualize_principle_components(df_proj=df_prolife_proj, cumulative_explained_variance=prolife_explained_variance,title="Pro-life subreddit discussions")
visualize_principle_components(df_proj=df_prochoice_proj, cumulative_explained_variance=prochoice_explained_variance, title="Pro-choice subreddit discussions")
visualize_principle_components(df_proj=df_proj, cumulative_explained_variance=cum_explained_variance, label_class=["Pro-life subreddits", "Pro-choice subreddits"], title="Combined subreddit discussions projections", cat_col='category')

visualize_3d_plot(df_proj, df_proj['category'])

prochoice_tsne_proj, _ = compute_proj_and_variance(prochoice_embed_matrix, "t-SNE", n_components=3)
prolife_tsne_proj, _ = compute_proj_and_variance(prolife_embed_matrix, "t-SNE", n_components=3)
df_proj_tsne, _ = compute_proj_and_variance(combined_embed_matrix, "t-SNE", n_components=3)

df_proj_tsne['category'] = cat_label.flatten()

visualize_tsne_components(df_proj=prochoice_tsne_proj)
visualize_tsne_components(df_proj=prolife_tsne_proj)
visualize_tsne_components(df_proj= df_proj_tsne, cat_col='category')

visualize_principle_components("t-SNE", prochoice_embed_matrix, n_components=10)
visualize_principle_components("t-SNE", prolife_embed_matrix, n_components=10)
visualize_principle_components("t-SNE", combined_embed_matrix, n_components = 10, color="red")


## Binary outcome versions 
prochoice_df['bin_num_comments'] = prochoice_df['num_comments'].apply(lambda x: convert_binary_output(x))
prolife_df['bin_num_comments'] = prolife_df['num_comments'].apply(lambda x: convert_binary_output(x))

df_prolife_proj['bin_comments'] = prolife_df['bin_num_comments']
df_prochoice_proj['bin_comments'] = prochoice_df['bin_num_comments']

visualize_principle_components(df_proj=df_prolife_proj, cumulative_explained_variance=prolife_explained_variance, label_class= ["No comments", "Had comments"] ,title="Pro-life subreddit discussions", cat_col='bin_comments')
visualize_principle_components(df_proj=df_prochoice_proj, cumulative_explained_variance=prochoice_explained_variance, label_class = ["No comments", "Had comments"], title="Pro-choice subreddit discussions", cat_col='bin_comments')


# Forward pass for a smaller subset of data 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

prochoice_df['category'] = pd.Series(np.zeros(len(prochoice_df)))
prolife_df['category'] = pd.Series(np.ones(len(prolife_df)))
combined_df = prochoice_df._append(prolife_df).reset_index(drop=True)
subset_df = combined_df[combined_df['WordCount'] <= 512]
subset_df['category'].value_counts()
# Articles with < or = 512 characters 
# 1.0    781
# 0.0    485
subset_df['forward_pass'] = subset_df['article'].apply(lambda x: forward_pass(model_name='bert-base-uncased',text= x))
subset_df = subset_df.reset_index(drop=True)


subset_df.to_pickle('combined_df_fwd_pass.pkl')
combined_df = pd.read_pickle('combined_df_fwd_pass.pkl')

stack_fwd_pass = np.vstack(combined_df['forward_pass'])

fwd_proj, explained_variance = compute_proj_and_variance(stack_fwd_pass, "PCA", 10)
fwd_proj['category'] = combined_df['category']

visualize_principle_components(fwd_proj, explained_variance, "analysis of ideologoically-charged articles surrounding abortion sentiments", label_class=["Pro-choice subreddits", "Pro-life subreddits"], cat_col='category')

index_collections = index_edge_cases(fwd_proj, combined_df, 'article', [0, 1])
index_collections.to_csv("index_collections_fwd_pass.csv")


"""Lasso and ridge regression to extract features that are important for explain key outcome variables"""

pooler_vector = np.vstack(combined_df['forward_pass'])
num_comments_fw =  pd.to_numeric(combined_df['num_comments'], errors='coerce').fillna(0).astype(int)
num_score_fw = pd.to_numeric(combined_df['score'], errors='coerce').fillna(0).astype(int)

param_grid_lasso = {
    'alpha': 
        [0.01, 0.1, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1, 1.2]
    }

ridge_param_grid = {
    'alpha': np.logspace(-3, 3, 13),          
    'solver': ['auto', 'lsqr'],
    'fit_intercept': [True, False]
}

apply_regularize_reg(method="lasso", data=pooler_vector, target=num_comments_fw, param_grid=param_grid_lasso, cv_num=5)
apply_regularize_reg(method="lasso", data=pooler_vector, target=num_score_fw, param_grid=param_grid_lasso, cv_num=5)
apply_regularize_reg(method="ridge", data=pooler_vector, target=num_comments_fw, param_grid=ridge_param_grid, cv_num=5)
apply_regularize_reg(method="ridge", data=pooler_vector, target=num_score_fw, param_grid=ridge_param_grid, cv_num=5)
## Params of interest 128, 256, 358, 521

# 128  0.015196
# 256  0.177495
# 358 -0.371344

df_params = pd.DataFrame(np.vstack(combined_df['forward_pass']))
df_interested_params = df_params[[128, 256, 358, 521]].copy()
df_int_params_proj, explained_variance= compute_proj_and_variance(df_interested_params, "PCA", 2)
df_int_params_proj['bin_com'] = combined_df['category']
visualize_principle_components(df_int_params_proj, explained_variance, title="", label_class=["choice", "life"], cat_col='bin_com')

## We can use NMDS to reconstruct the distances perhaps? 
params_dist = pdist(df_interested_params, metric='euclidean')
df_params_dist = pd.DataFrame((1-squareform(params_dist)))

nmds = MDS(n_components=3, metric=False, dissimilarity="precomputed", random_state=42, n_init=10, max_iter=300)
nmds_coords = nmds.fit_transform(df_params_dist)
nmds_coords = pd.DataFrame(nmds_coords)
nmds_coords['category'] = combined_df['category']

category_map = {
    0: "Pro-choice subreddits",
    1: "Pro-life subreddits"
}
nmds_coords['category'] = nmds_coords['category'].map(category_map)
colors = {"Pro-choice subreddits": 'steelblue', "Pro-life subreddits": 'darkorange'}


combined_df['embedding'] = combined_df['forward_pass'].apply(lambda x: np.squeeze(x))
df_embedding = pd.DataFrame(combined_df['embedding'].to_list(), columns=[f'neuron_{i}' for i in range(768)])
## Params of interest 128, 256, 358, 521

col_of_interest=[]
for num in [128, 256, 358, 521]: 
    col_of_interest.append(f'neuron_{num}')

edge_cases = index_edge_cases(df_embedding, combined_df, 'article', col_of_interest)
edge_cases.to_csv('export_df_for_arianna.csv')
combined_df

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for cat in nmds_coords['category'].unique():
    subset = nmds_coords[nmds_coords['category'] == cat]
    ax.scatter(subset[0], subset[1], subset[2], 
               color=colors[cat], label=f'{cat}', alpha=0.3)

ax.set_xlabel('NMDS1')
ax.set_ylabel('NMDS2')
ax.set_zlabel('NMDS3')
ax.set_title('3D NMDS ordination of relevant features that predict comments')
ax.legend()
plt.show()


## Scatter plot of log transformed ratings 

fwd_proj['score'] = pd.to_numeric(combined_df['score'], errors='coerce').fillna(0).astype(int)
fwd_proj['log_score'] = np.log(pd.to_numeric((combined_df['score']), errors='coerce').fillna(0).astype(int))

fwd_proj['num_comments'] = pd.to_numeric(combined_df['num_comments'], errors='coerce').fillna(0).astype(int)
fwd_proj['log_num_comments'] = np.log(pd.to_numeric(combined_df['num_comments'], errors='coerce').fillna(0).astype(int))

fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(fwd_proj[0], fwd_proj[1], c=fwd_proj['log_score'].values, cmap='plasma')
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('rating')  # label for the colorbar
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Scatter plot colored by log-transformed ratings')
plt.show()



"""--------------------------------------------------------------"""

"""Can we split and find a decision boundary that separates conservative and liberal articles or other binary tasks"""

param_grid = {
    'solver': ['svd', 'eigen'],
    'shrinkage': ['auto', 0.1, 0.5, 0.9],

}

"""
Perform LDA and evaluate performance on smaller dataset. 1266x768
X: pooler_vector
y: target_bin_vec_count (binary variable if there is a comment)
y: target_bin_count (binary variable for which subreddit article is extracted from)
y: target_bin_vec_score (binary variable for score of the dataframe)
"""
target_bin_vec_count = combined_df['bin_num_comments']
target_bin_vec = np.asarray(combined_df['category'])

apply_evaluate_LDA(pooler_vector, target_bin_vec_count, param_grid, cv_num=5)
## 0.6 in identify comments of smaller dataset 
apply_evaluate_LDA(pooler_vector, target_bin_vec, param_grid, cv_num=5)
## 0.66 in identify category (conservative or liberal)

# grid_lda_coeffs = pd.DataFrame(grid_lda.coef_).T
# grid_lda_coeffs[0].sort_values(ascending=False).head(20)


"""
Perform LDA and evaluate performance on larger dataset 5759x384

X: combined_transformer_embed-- BERT embedded vectors of articles 
y: cat_label: (binary variable for which subreddit article is extracted from)
y: bin_comment:  (binary variable if there is a comment)
y: bin_score: (binary variable if there is a score)
"""

combined_transformer_embed = pd.DataFrame(combined_embed_matrix)
combined_transformer_embed['category'] = cat_label

cat_label
bin_comment = combined_df_transformer_1['num_comments'].reset_index(drop=True).apply(lambda x: convert_binary_output(x))
bin_score = combined_df_transformer_1['score'].reset_index(drop=True).apply(lambda x: convert_binary_output(x))
apply_evaluate_LDA(combined_transformer_embed, cat_label, param_grid, 5)

## Linear discriminant analysis of pro-choice and pro-life categories

category_label = {
    "0": "Pro-choice",
    "1": "Pro-life"
}


best_lda, _, _, _, _ = apply_evaluate_LDA(combined_transformer_embed.drop(index=[3131, 3291, 1243, 5417, 3396, 270]), pd.DataFrame(cat_label).drop(index=[3131, 3291, 1243, 5417, 3396, 270]), param_grid, cv_num=5)
plot_linear_discriminants(combined_transformer_embed, cat_label, param_grid, cv_num=5, label_dict =category_label, plot_title="LDA projection of which articles are shared in conservative or liberal channels\n\nprojected data to the best discriminative axis")

coefficients = best_lda.coef_[0]
abs_coefficients = np.abs(coefficients)
sorted_indices = np.argsort(abs_coefficients)[::-1]

classify_ideology_index = index_edge_cases_3(combined_transformer_embed.drop(index=[1312, 695, 2690, 3131, 3133, 3291, 1243, 5417, 3396, 270, 830, 3949, 2967, 5573, 2740]), combined_df, "article", sorted_indices[0:3])
classify_ideology_index.to_csv('classify_ideology_index.csv')

classify_ideology_index[['min_1', 'min_2', 'min_3', 'median_1', 'median_2', 'median_3', 'max_1', 'max_2', 'max_3']]
classify_ideology_index[['min_text_1', 'min_text_2', 'min_text_3', 'median_text_1', 'median_text_2', 'median_text_3', 'max_text_1', 'max_text_2', 'max_text_3']]

## Binary outcome variable of commenting LDA projection

best_lda, _, _, _, _ = apply_evaluate_LDA(combined_transformer_embed, bin_comment.to_numpy(), param_grid, cv_num=5)

comment_dict = {
    "0": "no comment",
    "1": "comment"
}

plot_linear_discriminants(combined_transformer_embed, bin_comment.to_numpy().reshape(-1, 1), param_grid, 5, "LDA projection to discriminate likelihood of commenting", comment_dict, )

coefficients = best_lda.coef_[0]
# abs_coefficients = np.abs(coefficients)
coefficients = np.abs(coefficients)
sorted_indices = np.argsort(coefficients)[::-1]

classify_ideology_index = index_edge_cases_3(combined_transformer_embed, combined_df, "article", sorted_indices[0:3])

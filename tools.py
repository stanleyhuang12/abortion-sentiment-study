from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, classification_report, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import torch 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from transformers import BertTokenizer, BertModel


def visualize_3d_plot(data, category): 
    plt.figure()
    
    ax = plt.axes(projection="3d")
    
    ax.scatter3D(data[0], data[1], data[2], c=category, cmap="viridis", alpha=0.6)
    
    ax.set_xlabel("First dimension")
    ax.set_ylabel("Second dimension")
    ax.set_zlabel("Third dimension")
   
    plt.legend()
    plt.show()

def compute_proj_and_variance(matrix, method, n_components): 
    if method == "PCA":
        pca_model = PCA(n_components=n_components)
        model_projections = pca_model.fit_transform(matrix)
        cumulative_explained_variance = np.cumsum(pca_model.explained_variance_)
    if method == "t-SNE":
        tsne_model = TSNE(n_components=n_components)
        model_projections = tsne_model.fit_transform(matrix)
        cumulative_explained_variance = None
    df_proj = pd.DataFrame(model_projections)
    print(f"Cumulative sum of explained variance of {n_components} using {method} method: {cumulative_explained_variance}") 

    return df_proj, cumulative_explained_variance


def visualize_principle_components(df_proj, cumulative_explained_variance, title, label_class, cat_col=None): 
    figs, ax = plt.subplots(ncols=2)
    figs.suptitle("Projected components and cumulative explained variances")
    
    df_proj_plot = df_proj.iloc[:, 0:2]
    
    if cat_col is not None:
        df_proj_plot[cat_col] = df_proj[cat_col]
        sub_1 = df_proj_plot.loc[df_proj_plot[cat_col] == 0]
        sub_2 = df_proj_plot.loc[df_proj_plot[cat_col] == 1]
        ax[0].scatter(sub_1.iloc[:, 0], sub_1.iloc[:, 1], c="blue", label = label_class[0], alpha = 0.6)
        ax[0].scatter(sub_2.iloc[:, 0], sub_2.iloc[:, 1], c="red", label= label_class[1], alpha = 0.6)
        ax[0].legend(title="Category")
    else: 
        ax[0].scatter(df_proj_plot[0], df_proj_plot[1], alpha = 0.6)
        
    ax[0].set_title(f"Projected Principle Components {title}")

    ax[0].set_xlabel("X dimension")
    ax[0].set_ylabel("Y dimension")
    
    ## Plot cumulative explained variance (either PCA or TSNE method)
    ax[1].plot(cumulative_explained_variance)
    ax[1].set_xlabel("Number of components")
    ax[1].set_ylabel('Cumulative explained variance')
    ax[1].grid(True)
        
    plt.tight_layout()
    plt.show()

def visualize_tsne_components(df_proj, cat_col=None): 
    figs, ax = plt.subplots()
    figs.suptitle("Projected components and cumulative explained variances")
    
    df_proj_plot = df_proj.iloc[:, 0:2]
    
    if cat_col is not None:
        df_proj_plot[cat_col] = df_proj[cat_col]
        sub_1 = df_proj_plot.loc[df_proj_plot[cat_col] == 0]
        sub_2 = df_proj_plot.loc[df_proj_plot[cat_col] == 1]
        ax.scatter(sub_1.iloc[:, 0], sub_1.iloc[:, 1], c="blue", label = "Pro-choice subreddit", alpha = 0.6)
        ax.scatter(sub_2.iloc[:, 0], sub_2.iloc[:, 1], c="red", label="Pro-life subreddit", alpha = 0.6)
        ax.legend(title="Category")
    else: 
        ax.scatter(df_proj_plot[0], df_proj_plot[1], alpha = 0.6)
    ax.set_title(f"Projected Principle Components")

    ax.set_xlabel("X dimension")
    ax.set_ylabel("Y dimension")
    
    plt.show()


def forward_pass(model_name, text): 
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)
    
    tokenized_texts = tokenizer(text, truncation=True, return_tensors = 'pt')
    bert_model.eval()
    
    with torch.no_grad(): 
        model_outputs = bert_model(**tokenized_texts)
        return model_outputs.pooler_output
        

## Takes in the pooler outputs and uses a linear regression task to predict score and predict number of comments 

def index_edge_cases_3(df_embedding, df_article, article_col, col_list):
    df_embedding = df_embedding.copy()
    
    # Convert the embeddings to numerical values if necessary
    for col in col_list:
        df_embedding[col] = df_embedding[col].apply(lambda x: x.item() if hasattr(x, 'item') else x)
    
    # Prepare a dataframe to store the index collections for min, max, and median
    index_collections = pd.DataFrame(index=col_list, columns=['min_1', 'min_2', 'min_3', 'max_1', 'max_2', 'max_3', 'median_1', 'median_2', 'median_3', 
                                                              'min_text_1', 'min_text_2', 'min_text_3', 'max_text_1', 'max_text_2', 'max_text_3',
                                                              'median_text_1', 'median_text_2', 'median_text_3'])  
    for feature in col_list:
        # Get the indices of the top 3 minimum and maximum values
        min_indices = df_embedding[feature].nsmallest(3).index
        max_indices = df_embedding[feature].nlargest(3).index
        
        # Get the median value and find the closest 3 indices
        median_val = df_embedding[feature].median()
        median_indices = (df_embedding[feature] - median_val).abs().nsmallest(3).index
        
        # Store the indices and corresponding article texts in the dataframe
        index_collections.loc[feature, 'min_1'] = min_indices[0]
        index_collections.loc[feature, 'min_2'] = min_indices[1]
        index_collections.loc[feature, 'min_3'] = min_indices[2]
        index_collections.loc[feature, 'min_text_1'] = df_article[article_col].iloc[min_indices[0]]
        index_collections.loc[feature, 'min_text_2'] = df_article[article_col].iloc[min_indices[1]]
        index_collections.loc[feature, 'min_text_3'] = df_article[article_col].iloc[min_indices[2]]
        
        index_collections.loc[feature, 'max_1'] = max_indices[0]
        index_collections.loc[feature, 'max_2'] = max_indices[1]
        index_collections.loc[feature, 'max_3'] = max_indices[2]
        index_collections.loc[feature, 'max_text_1'] = df_article[article_col].iloc[max_indices[0]]
        index_collections.loc[feature, 'max_text_2'] = df_article[article_col].iloc[max_indices[1]]
        index_collections.loc[feature, 'max_text_3'] = df_article[article_col].iloc[max_indices[2]]
        
        index_collections.loc[feature, 'median_1'] = median_indices[0]
        index_collections.loc[feature, 'median_2'] = median_indices[1]
        index_collections.loc[feature, 'median_3'] = median_indices[2]
        index_collections.loc[feature, 'median_text_1'] = df_article[article_col].iloc[median_indices[0]]
        index_collections.loc[feature, 'median_text_2'] = df_article[article_col].iloc[median_indices[1]]
        index_collections.loc[feature, 'median_text_3'] = df_article[article_col].iloc[median_indices[2]]
    
    return index_collections


def index_edge_cases(df_embedding, df_article, article_col, col_list):
    """Passes in a dataframe of embedding, a dataframe with article text, a column list of relevant neurons, outputs index of articles worth looking into"""
    df_embedding = df_embedding.copy()
    
    for col in col_list:
        df_embedding[col] = df_embedding[col].apply(lambda x: x.item() if hasattr(x, 'item') else x)
    
    
    index_collections = pd.DataFrame(index=col_list, columns=['min', 'max', 'median', 'min_text', 'max_text', 'median_text'])
    
    for feature in col_list: 
        maximum_idx = df_embedding[feature].idxmax()
        minimum_idx = df_embedding[feature].idxmin()
        
        median_val = df_embedding[feature].median()
        median_idx = (df_embedding[feature] - median_val).abs().idxmin()

        index_collections.loc[feature, 'min'] = minimum_idx
        index_collections.loc[feature, 'min_text'] = df_article[article_col].iloc[minimum_idx]
        
        index_collections.loc[feature, 'max'] = maximum_idx
        index_collections.loc[feature, 'max_text'] = df_article[article_col].iloc[maximum_idx]
        
        index_collections.loc[feature, 'median'] = median_idx
        index_collections.loc[feature, 'median_text'] = df_article[article_col].iloc[median_idx]
        
    
    return index_collections

        
def apply_regularize_reg(method, data, target, param_grid, cv_num):
    """Apply a regularized linear regression method."""
    if method == "lasso": 
        model = Lasso()
    elif method == "ridge": 
        model = Ridge()
        
    grid_search_model = GridSearchCV(model, param_grid, cv=cv_num)
    
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=21)
    
    grid_search_model.fit(X_train, y_train)
    print("Grid Search finds the best parameters are: ", grid_search_model.best_params_)

    best_model = grid_search_model.best_estimator_

    y_pred_b = best_model.predict(X_test)
    
    y_pred_b = best_model.predict(X_test)
    
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_b))
    print("Mean Squared Error:\n", mean_squared_error(y_test, y_pred_b))
    print("R2 Score\n", r2_score(y_test, y_pred_b))
    



def apply_evaluate_LDA(data, target, param_grid, cv_num):
    lda = LinearDiscriminantAnalysis(n_components=1)
    grid_search_lda = GridSearchCV(lda, param_grid, cv=cv_num)
    
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=21)
    
    grid_search_lda.fit(X_train, y_train)
    print("Grid Search finds the best parameters are: ", grid_search_lda.best_params_)
    
    best_lda = grid_search_lda.best_estimator_
    best_lda.fit(X_train, y_train)
    y_pred_b = best_lda.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred_b))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_b))
    print("Classification Report:\n", classification_report(y_test, y_pred_b))

    return best_lda, X_train, X_test, y_train, y_test
    
def plot_linear_discriminants(x, y, param_grid, cv_num, plot_title, label_dict = None):
    
    best_lda, X_train, X_test, y_train, y_test = apply_evaluate_LDA(x, y, param_grid, cv_num)
    
    x_lda = np.vstack((X_train, X_test))
    y_lda = np.vstack((y_train, y_test))
    
    projected_data = best_lda.transform(x_lda)
    
    
    if projected_data.shape[1] == 1:
        # Only 1 dimension, plot along X-axis
        plt.scatter(projected_data[y_lda == 0], np.zeros_like(projected_data[y_lda == 0]), label=f'Class {label_dict.get("0")}', alpha=0.2)
        plt.scatter(projected_data[y_lda == 1], np.ones_like(projected_data[y_lda == 1]), label=f'Class {label_dict.get("1")}', alpha=0.2)

        mean_0 = np.mean(projected_data[y_lda==0])
        mean_1 = np.mean(projected_data[y_lda==1])
        
        decision_line = np.mean((mean_0, mean_1))
        plt.axvline(decision_line, ymin=-0.5, ymax=1.5, linestyle="--", color='black')
        
        ## Brute force decision line
        decision_line = np.mean((mean_0, mean_1))
        plt.scatter(mean_0, 0, color='green', edgecolor='black')
        plt.text(mean_0, 0.12, f"Center of the \n in {label_dict.get("0")} subreddits ")
        plt.scatter(mean_1, 1, color='green', edgecolor='black')
        plt.text(mean_1-1.2, 0.88, f"Center of the articles  \n in {label_dict.get("1")} subreddits")

        plt.xlabel('LD1')
        plt.ylabel('')
    else:
        # 2+ dimensions, plot LD1 vs LD2
        for label in np.unique(y_lda):
            plt.scatter(projected_data[y_lda == label], projected_data[y_lda == label_dict.get(label)], label=f'Class {label}', alpha=0.2, edgecolors='none')
        plt.xlabel('LD1')
        plt.ylabel('LD2')
    
    plt.margins(x=0.1, y=0.1)
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def convert_binary_output(num): 
    try:
        return 0 if np.isclose(float(num), 0) else 1
    except (TypeError, ValueError):
        return 0 
    
## Deprecated 

def classify_pooler_outputs(df, col_pooler_outputs, col_target):
    """Takes a DataFrame of pooler outputs with target labels and trains a Linear Discriminant Analysis model 
    to identify what is the best projection to classify groups."""
   
    pooler_vector = df.loc[col_pooler_outputs]
    target = df[col_target] 
    
    ## Standardize pooler outputs: 
    sc = StandardScaler()
    sd_pooler_vector = sc.fit_transform(pooler_vector)
    
    X_train, y_train, x_test, y_test = train_test_split(sd_pooler_vector, target, split=0.2, stratify=target)
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)
    
    
    classifier = RandomForestClassifier(max_depth=2,
                                        random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    print('Accuracy : ' + str(accuracy_score(y_test, y_pred)))
    conf_m = confusion_matrix(y_test, y_pred)
    print(conf_m)
    
    return X_test



def regress_pooler_outputs(model, df, df_target): 
    pooler_vectors = np.vstack(df)
    
    target_vector = np.asarray(np.float64(df_target))
    
    sd_pooler_vectors = StandardScaler().fit_transform(pooler_vectors)
    
    X_train, X_test, y_train, y_test = train_test_split(sd_pooler_vectors, 
                                                        target_vector, 
                                                        test_size=0.3, 
                                                        random_state=15)
    
    lr_model = model
    lr_model.fit(X_train, y_train)
    
    y_pred = lr_model.predict(X_test)
    print(y_pred)
    
    return {
                "intercept": lr_model.intercept_, 
                "params": lr_model.coef_,
                "mse": mean_squared_error(y_pred, y_test)
            }
    

def cross_val_regression(model, df, df_target, k=5): 
    pooler_vectors = np.vstack(df)
    target_vector = np.asarray(df_target.values)

    sd_pooler_vectors = StandardScaler().fit_transform(pooler_vectors)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    mses = []
    coefs = []
    intercepts = []

    for train_index, test_index in kf.split(sd_pooler_vectors):
        X_train, X_test = sd_pooler_vectors[train_index], sd_pooler_vectors[test_index]
        y_train, y_test = target_vector[train_index], target_vector[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mses.append(mean_squared_error(y_test, y_pred))
        coefs.append(model.coef_)
        intercepts.append(model.intercept_)

    return {
        "mean_mse": np.mean(mses),
        "std_mse": np.std(mses),
        "avg_params": np.mean(coefs, axis=0),
        "avg_intercept": np.mean(intercepts)
    }

from itertools import count
import json
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

import pyLDAvis.sklearn

def createDocTermMatrix(corpus):
    """Creates a document-term matrix from a json csv containing documents

    Args:
        corpus (list): corpus containing documents of text where each string is a document

    Returns:
        matrix: document term matrix
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    document_term_matrix = vectorizer.fit_transform(corpus.astype('U'))
    return document_term_matrix, vectorizer

def trainTopicModel(document_term_matrix):
    """trains a topic model given a document term matrix
    https://www.analyticsvidhya.com/blog/2021/05/topic-modelling-in-natural-language-processing/

    Args:
        document_term_matrix (pandas.DataFrame): bag of words for topic modelling
    """
    # Parameters tuning using Grid Search
    grid_params = {'n_components' : [5, 20, 100]}
    # LDA model
    lda = LatentDirichletAllocation()
    lda_model = GridSearchCV(lda,param_grid=grid_params)
    lda_model.fit(document_term_matrix)
    # Estimators for LDA model
    lda_model1 = lda_model.best_estimator_
    print("Best LDA model's params" , lda_model.best_params_)
    print("Best log likelihood Score for the LDA model",lda_model.best_score_)
    print("LDA model Perplexity on train data", lda_model1.perplexity(document_term_matrix))

    return lda_model1

def visualiseTopics(lda_model, document_term_matrix, count_vectorizer):
    """visualises a trained model
    https://www.analyticsvidhya.com/blog/2021/05/topic-modelling-in-natural-language-processing/
    Args:
        lda_model (sklearn.decomposition.LatentDirichletAllocation): trained topic model
    """
    import pyLDAvis.sklearn
    prepared_data = pyLDAvis.sklearn.prepare(lda_model, document_term_matrix, count_vectorizer, mds='mmds')
    # pyLDAvis.show(prepared_data)
    pyLDAvis.save_html(prepared_data, 'topicsVis.html')


if __name__ == '__main__':
    createDocTermMatrix()
    trainTopicModel()
    visualiseTopics()
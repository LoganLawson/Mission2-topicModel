from extractReviews import extractTextOnly
from topicModelling import createDocTermMatrix, trainTopicModel, visualiseTopics

print('\n\nExtracting text...')
corpus = extractTextOnly('exampleData.json', limit=1000)
print(corpus[0:5])

print('\n\nCreating document term matrix...')
document_term_matrix, count_vectorizer= createDocTermMatrix(corpus)
print(document_term_matrix[0:5])

print('\n\nTraining topic model...')
lda_model = trainTopicModel(document_term_matrix)
print(lda_model)

visualiseTopics(lda_model, document_term_matrix, count_vectorizer)
print('\n\nVisualisation available at ./topicsVis.html')
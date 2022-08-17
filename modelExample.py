import requests
from extractReviews import extractTextOnly
from topicModelling import createDocTermMatrix, trainTopicModel, visualiseTopics

print('\n\nGetting some example data:')
res = requests.get('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Automotive_5.json.gz')
open('exampledata.json', 'wb').write(res.content)

print('\n\nExtracting text:')
corpus = extractTextOnly('exampledata.json', limit=1000)
print(corpus[0:5])

print('\n\nCreating document term matrix:')
document_term_matrix, count_vectorizer= createDocTermMatrix(corpus)
print(document_term_matrix)

print('\n\nTraining topic model:')
lda_model = trainTopicModel(document_term_matrix)
print(lda_model)

print('\n\nVisualisation available at ./topicsVis.html:')
visualiseTopics(lda_model, document_term_matrix, count_vectorizer)
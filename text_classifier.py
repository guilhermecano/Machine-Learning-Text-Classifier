'''

'''
#Importação dos pacotes
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

loader = load_files('filesk')
stemmer = SnowballStemmer("english")
y = loader.target

#Função para adição de stemming:
def preprocessador(data):
        return " ".join([SnowballStemmer("english").stem(word) for word in data.split()])

#Vetorizador para bag of words binário
bin_cv = CountVectorizer(strip_accents='ascii', stop_words='english', preprocessor = preprocessador, min_df=1, binary=True)

#Vetorizador contador (Não binário)
cv = CountVectorizer(strip_accents='ascii', stop_words='english', preprocessor = preprocessador, min_df=1, binary=False)

X_bin = bin_cv.fit_transform(loader.data) #Matriz binária
X_count = cv.fit_transform(loader.data) #Matriz com contagem de frequência

#Aplicação da transformada tf_idf:
tf_transformer = TfidfTransformer(use_idf=True)
X_tf = tf_transformer.fit_transform(X_count)

#Divisão do dataset em conjuntos de treino e teste, para ambos os casos:
sss = StratifiedShuffleSplit(test_size=0.2)

for train_index, test_index in sss.split(X_bin, y):
    X_treino_bin, X_teste_bin = X_bin[train_index], X_bin[test_index]
    y_treino, y_teste= y[train_index], y[test_index]

for train_index, test_index in sss.split(X_tf, y):
    X_treino_tf, X_teste_tf = X_tf[train_index], X_tf[test_index]

#Aplicação do naive Bayes:
nb = GaussianNB()
ypred = nb.fit(X_treino_bin.toarray(), y_treino).predict(X_teste_bin.toarray())
print("Acurácia média do Naive Bayes: %.3f"%nb.score(X_teste_bin.toarray(), y_teste))

#Uso da regressão Logística para os dois casos:
rl_bin = LogisticRegression(C=10000)
rl_tf = LogisticRegression(C=10000)
ypred_rlb = rl_bin.fit(X_treino_bin.toarray(), y_treino).predict(X_teste_bin.toarray())
ypred_rltf = rl_tf.fit(X_treino_tf.toarray(), y_treino).predict(X_teste_tf.toarray())

print("Acurácia media da regressão logística para as matrizes:")
print("Bag of word binário: %.3f"%rl_bin.score(X_teste_bin.toarray(), y_teste))
print("Term frequence: %.3f"%rl_tf.score(X_teste_tf.toarray(), y_teste))



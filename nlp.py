from nltk.stem import SnowballStemmer
from math import log

def stem(tokenized_corpus):
	"""Réalise le stemming des commentaires en utilisant le SnowballStemmer de NLTK.

	ENTREE
	----------
	tokenized_corpus : array de strings
		Le tableau de commentaires tokenizé.

	SORTIE
	----------
	stemmed_tab : array de strings
		Le tableau de commentaires racinisé.
	"""

	stemmer = SnowballStemmer("english")

	stemmed_corpus = []

	for comment in  tokenized_corpus:
		tab = []
		for word in comment:
			tab.append(stemmer.stem(word))

		stemmed_corpus.append(tab)

	return stemmed_corpus


def tfidf(tokenized_corpus, word, comment, df_dict):
	"""Retourne la TF-IDF de word dans comment."""

	n = len(tokenized_corpus)
	tf = comment.count(word)
	df = df_dict[word]

	tfidf = tf*log(n/df,2)

	return tfidf
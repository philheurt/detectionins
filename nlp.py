from nltk.stem import SnowballStemmer
from nltk import word_tokenize, sent_tokenize
from math import log

import string
import re


def tokenize(corpus, auto=True, sentence=False):
	"""Sépare chaque string du tableau en un tableau de ses mots.
	Problème de NLTK : ne sépare pas les mots séparés uniquement par un slash (au moins ça).

	ENTREE
	-----------
	corpus : liste de strings
		Le tableau de commentaires

	auto : booléen
		Si True, le module nltk sera utilisé au lieu de mon implémentation (pourtant admirable).
		La ponctuation apparaîtra alors dans le tableau de commentaires tokenizé.

	sentence : booléen
		Si True, utilse la fonction sent_tokenize de nltk qui sépare en plus les commentaires en phrases.
		2 phrases -> 2 tableaux des phrases tokenizées.

	SORTIE
	-----------
	tokenized_corpus : liste d'listes de strings
		Le tableau des commentaires tokenizé
	"""

	tok_tab = []
	if auto:
		for comment in corpus:
			if sentence:
				tok_tab.append(sent_tokenize(comment))
			else:
				tok = word_tokenize(comment)
				for word in tok:
					if '/' in word: # le tokenizer de nltk ne sépare pas les mots séparé par un slash sans espace
						tok.extend(word.split('/')) # l'ordre des mots n'a pas d'importance donc on rajoute à la fin
						tok.remove(word)
				tok_tab.append(tok)
	else:
		for comment in corpus:

			if comment[0] in string.punctuation:
				tok = re.split('\W+',comment)[1:]
			else:
				tok = re.split('\W+',comment)

			if len(tok) > 1:
				tok_tab.append(tok[:-1])
			else:
				tok_tab.append(tok)

	return tok_tab


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
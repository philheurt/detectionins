import numpy as np
import pandas as pd

import string
import re
import nltk

def extract_comments(filename, test=False):
	"""Récupère les commentaires ainsi que leur label si train=True (défaut).

	ENTREE
	-----------
	filename : string
		Le nom du fichier

	test : booleen
		Si test vaut True, la fonction renvoit la variable cible. Sinon elle renvoit seulement les commentaires.

	SORTIE
	----------
	X : array de strings
		Le tableau de commentaire

	y : array de booleens
		La variable cible renvoyée si test=False.
	"""
	X=[]
	with open(filename, "r") as f:
		if not test:
			y=[]
			for line in f:
				y.append(int(line[0]))
				X.append(line[5:-4])

			y = np.array(y)
			return X, y
		else:
			for line in f:
				X.append(line[1:-1])
				
			return X


def clean(stringtab):
	"""Nettoie les commentaires et remplace les éléments indésirables par des balises adaptées.
	Largement inspiré de https://github.com/andreiolariu/kaggle-insults/blob/master/nlp_dict.py
	
	ENTREE
	----------
	stringtab : array de strings
		Le tableau de commentaires

	SORTIE
	----------
	tab : array de strings
		Tableau de commentaire modifié
	"""

	# signes d'encodage et balises html
	regex_enc_tag = re.compile(r"\\r|\\n|\\x(\w{2})|\\u[A-F\d]{4}|<[^>]*>")
	# adresses url
	regex_url = re.compile(r"https?://[^\s]*")
	# limite le nombre de lettres identiques consécutives à 2
	regex_reps = re.compile(r"([a-zA-Z])\1\1+(\w*)")
	# nombres seuls, a priori pas indicatifs du caractère insultant
	regex_nbs = re.compile(r"\s\d+[.!?,;:\"\s]")
	# dates, on peut penser que certaines dates sont importantes (comme 9/11) mais je pense qu'elles ne disent pas grand chose du caractère insultant
	regex_date = re.compile(r"\s\d?\d[./\-]\d?\d[./-]\d\d\d\d[.!?,;:\"\s]") # ne vérifie pas que les valeurs sont plausibles mais devrait suffire
	# recompose les mots dont les lettres sont séparées par des points ou des tirets
	regex_dotted_words = re.compile(r"[.\-\s](\w)[.-](\w)") # pour ne pas confondre les acronymes ou tentative de dissimulation de mots avec de la ponctutation
	# pour les personnes qui jurent comme dans une BD
	regex_curse = re.compile(r"([a-zA-Z]*)[#%&\*\$]{2,}([a-zA-Z]*)")
	# interprète les smileys
	regex_happy = re.compile(r"\^\^|[B8:;=xX]-?([)DpP]|\)\))")
	regex_sad = re.compile(r":'\(|[:=]-?[S(/$]")
	regex_ang = re.compile(r"-_-|:@|:-?\|")

	tab = []
	for comment in stringtab:

		no_enc_tag = regex_enc_tag.sub(" ", comment)
		no_url = regex_url.sub(" _url ", no_enc_tag)
		no_nbs = regex_nbs.sub(" _nb ", no_url)
		no_date = regex_date.sub(" _date ", no_nbs)
		no_dotted_words = regex_dotted_words.sub(r" \1\2 ", no_date)
		no_reps = regex_reps.sub(r" \1\1\2 _rep ", no_dotted_words)
		no_curse = regex_curse.sub(" fuck ", no_reps) # remplacer par fuck ou par une balise spéciale ? fuck dans bad_words au moins
		no_happy = regex_happy.sub(" _happysmiley ", no_curse)
		no_sad = regex_sad.sub(" _sadsmiley ", no_happy)
		no_ang = regex_ang.sub(" _angrysmiley ", no_sad)

		tab.append(no_ang)

	return tab


def tokenize(stringtab, auto=True, sentence=False):
	"""Sépare chaque string du tableau en un tableau de ses mots.
	Problème de NLTK : ne sépare pas les mots séparés uniquement par un slash (au moins ça)

	ENTREE
	-----------
	stringtab : array de strings
		Le tableau de commentaires

	auto : booléen
		Si True, le module nltk sera utilisé au lieu de mon implémentation (pourtant admirable).
		La ponctuation apparaîtra alors dans le tableau de commentaires tokenizé.

	sentence : booléen
		Si True, utilse la fonction sent_tokenise de nltk qui sépare en plus les commentaires en phrases.
		2 phrases -> 2 tableaux des phrases tokenizées.

	SORTIE
	-----------
	tokenized_tab : array d'arrays de strings
		Le tableau des commentaires tokenizé
	"""

	tok_tab = []
	if auto:
		for comment in stringtab:
			if sentence:
				tok_tab.append(nltk.sent_tokenize(comment))
			else:
				tok_tab.append(nltk.word_tokenize(comment))
	else:
		for comment in stringtab:

			if comment[0] in string.punctuation:
				tok = re.split('\W+',comment)[1:]
			else:
				tok = re.split('\W+',comment)

			if len(tok) > 1:
				tok_tab.append(tok[:-1])
			else:
				tok_tab.append(tok)

	return tok_tab


def count_punctuation(stringtab, ratio=True, auto=True):
	"""Retourne des informations sur le nombre de signes de ponctuation dans chaque commentaire.
	
	ENTREE
	------------
	stringtab : array de strings
		Le tableau de commentaires (nettoyé normalement), tokenizé ou non selon la valeur de auto.

	ratio : booléen
		Si on veut une valeur absolue ou relative à la taille du commentaire

	auto : booléen
		Si l'on a utilisé nltk dans tokenize. La ponctuation apparaît alors dans le tableau tokenizé.

	SORTIE
	------------
	cnt_tab : numpy.ndarray
		Tableau contenant pour chaque commentaire le nombre de signe de ponctuation, divisé ou non, selon la valeur de ratio,
		par la longueur du commentaire (en nombre de mots si auto=True, en nombre de caractères sinon).
	"""
	
	cnt_tab = []
	if auto:
		for comment in stringtab:
			cnt = 0
			for word in comment:
				if word in string.punctuation:
					cnt += 1
			if ratio:
				cnt_tab.append(cnt/len(comment))
			else:
				cnt_tab.append(cnt)

	else:
		for comment in stringtab:
			comment_no_space = [c for c in comment if c not in " _'"] #  on ajoute des espaces et _REP dans clean donc on ne les considère pas
			cnt = 0
			for c in comment_no_space:
				if c in string.punctuation:
					cnt += 1
			if ratio:
				cnt_tab.append(cnt/len(comment_no_space))
			else:
				cnt_tab.append(cnt)

	cnt_tab = np.array(cnt_tab)

	return cnt_tab


def count_capitals(tokenized_tab, ratio=True):
	"""Compte le nombre absolu ou relatif de lettres majuscules dans le tableau de commentaires tokenizé.
	
	ENTREE
	------------
	tokenized_tab : array 2d de strings
		Le tableau des commentaires tokenisés

	ratio : booléen
		Si on veut une valeur absolue ou relative à la taille du commentaire

	SORTIE
	------------
	tokenized_tab_low : array 2d de strings
		Copie de tokenized_tab mais avec tous les mots en minuscules

	cnt_tab : numpy.ndarray
		Tableau contenant pour chaque commentaire le nombre de lettres en majuscule, divisé ou non, selon la valeur de ratio,
		par la longueur du commentaire.
	"""

	cnt_tab = []
	tokenized_tab_low = []
	for tokenized_comment in tokenized_tab:
		cnt_cap = 0
		cnt = 0
		tab = []
		for word in tokenized_comment:
			for c in word:
				if c == c.upper() and c not in string.punctuation:
					cnt_cap += 1
				cnt += 1
			tab.append(word.lower())

		tokenized_tab_low.append(tab)

		if ratio:
			cnt_tab.append(cnt_cap/cnt)
		else:
			cnt_tab.append(cnt_cap)

	cnt_tab = np.array(cnt_tab)

	return tokenized_tab_low, cnt_tab


def load_bad_words():
	"""Retourne la liste des bad words établie par Google.

	SORTIE
	------------
	tab : array de strings
		La liste des bad words sous la forme d'un tableau.
	"""
	bw = open("bad_words.txt", "r")
	s = bw.read()
	bad_tab = re.split("\W+", s)
	# le premier et dernier élément de la liste sont vides (conséquence du split, les premier et dernier caractères du fichier étant des guillemets)
	return bad_tab[1:-1]


def get_bw_stats(tokenized_tab, ratio=True):
	"""Retourne le ratio de bad_words contenus dans les commentaires

	ENTREE
	-------------
	tokenized_tab : array de strings
		Le tableau de commentaires tokenizé.

	ratio : booléen
		True si on veut une valeur relative, False si absolue.

	SORTIE
	-------------
	bw_cnt : numpy.ndarray
		Le tableau contenant le ratio de chaque commentaire.
	"""

	bad_words = load_bad_words()

	bw_cnt = []
	for comment in tokenized_tab:
		comment_no_punct = [word for word in comment if word not in string.punctuation]
		cnt = 0
		for word in comment_no_punct:
			if word in bad_words:
				cnt += 1
		if ratio:
			#bw_cnt.append({'cnt_bw' : cnt, 'tot' : len(comment_no_punct)})
			bw_cnt.append(cnt/len(comment_no_punct))
		else:
			bw_cnt.append(cnt)

	bw_cnt = np.array(bw_cnt)

	return bw_cnt

def get_statistics(tokenized_tab, ratio=True):
	"""Retourne les nombres ou ratio de :
	- signes de ponctuation ;
	- majuscules ;
	- bad words.
	A n'utiliser que si l'on a pris auto=True auparavant.
	
	ENTREE
	------------
	tokenized_tab : array 2d de strings
		Le tableau des commentaires tokenisés

	ratio : booléen
		Si on veut une valeur absolue ou relative à la taille du commentaire

	SORTIE
	------------
	tokenized_tab_low = array de strings
		Le tableau de commentaires tokenizé et tout en minuscules.

	maj_stats : numpy.ndarray
		Le tableau de statistiques sur les majuscules.

	punct_stats : numpy.ndarray
		Le tableau de statistiques sur la ponctuation.

	bw_stats : numpy.ndarray
		Le tableau de statistiques sur les bad words.
	"""

	tokenized_tab_low, maj_stats = count_capitals(tokenized_tab, ratio)
	punct_stats = count_punctuation(tokenized_tab, ratio)
	bw_stats = get_bw_stats(tokenized_tab, ratio)

	return tokenized_tab_low, maj_stats, punct_stats, bw_stats


def get_features(tokenized_tab):
	"""Retourne différentes données sur le contenu des commentaires.

	ENTREE
	----------
	tokenized_tab : array de strings
		Le tableau des commentaires tokenizé

	SORTIE
	----------
	df : pandas DataFrame
		DataFrame contenant des infos numériques sur les commentaires
	"""
	n = len(tokenized_tab)
	lengths = []

	keys = ['_url', '_nb', '_date', '_rep', '_happysmiley', '_sadsmiley', '_angrysmiley']

	features = {}
	for key in keys:
		features[key] = np.zeros(n)

	for i,com in enumerate(tokenized_tab):
		length = 0
		for word in com:
			length += len(word) # ne marche que si l'on a pas tokenizé par phrase
			if word in keys:
				features[word][i] += 1

		lengths.append(length)

	lengths = np.array(lengths)

	df = pd.DataFrame(features)

	new_keys = ['nb_url', 'nb_nb', 'nb_date', 'nb_rep', 'nb_hsm', 'nb_ssm', 'nb_asm']
	df.columns = new_keys

	_, maj_stats, pun_stats, bw_stats = get_statistics(tokenized_tab)

	df['length'] = lengths
	df['maj_ratio'] = maj_stats
	df['punct_ratio'] = pun_stats
	df['bw_ratio'] = bw_stats

	return df
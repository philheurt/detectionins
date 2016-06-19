import numpy as np
import string
import re

def extract_features(filename, train=False):
	"""Récupère les commentaires ainsi que leur label si train=True (défaut).

	ENTREE
	-----------
	filename : string
		Le nom du fichier

	train : booleen
		Si train vaut True, la fonction renvoit la variable cible. Sinon elle renvoit seulement les commentaires.

	SORTIE
	----------
	X : array de strings
		Le tableau de commentaire

	y : array de booleens
		La variable cible renvoyée si train=False.
	"""
	X=[]
	with open(filename, "r") as f:
		if not train:
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
	regex_curse = re.compile(r"[#%&\*\$]{2,}")
	# interprète les smileys
	regex_happy = re.compile(r"\^\^|[B8:;=xX]-?([)DpP]|\)\))")
	regex_sad = re.compile(r":'\(|[:=]-?[S(/$]")
	regex_ang = re.compile(r"-_-|:@|:-?\|")

	tab = []
	for comment in stringtab:

		no_enc_tag = regex_enc_tag.sub(" ", comment)
		no_url = regex_url.sub(" _URL ", no_enc_tag)
		no_nbs = regex_nbs.sub(" _NB ", no_url)
		no_date = regex_date.sub(" _DATE ", no_nbs)
		no_dotted_words = regex_dotted_words.sub(r" \1\2 ", no_date)
		no_reps = regex_reps.sub(r" \1\1\2 _REP ", no_dotted_words)
		no_curse = regex_curse.sub(" fuck ", no_reps) # remplacer par fuck ou par une balise spéciale ? fuck dans bad_words au moins
		no_happy = regex_happy.sub(" _HAPPY_SMILEY ", no_curse)
		no_sad = regex_sad.sub(" _SAD_SMILEY ", no_happy)
		no_ang = regex_ang.sub(" _ANGRY_SMILEY ", no_sad)

		tab.append(no_ang)

	return tab

def count_punctuation(stringtab, ratio=True):
	"""Retourne des informations sur le nombre de signes de ponctuation dans chaque commentaire.
	
	ENTREE
	------------
	stringtab : array de strings
		Le tableau de commentaires (nettoyé normalement)

	ratio : booléen
		Si on veut une valeur absolue ou relative à la taille du commentaire

	SORTIE
	------------
	cnt_tab : array de ints ou de floats
		Tableau contenant pour chaque commentaire le nombre de signe de ponctuation, divisé ou non, selon la valeur de ratio,
		par la longueur du commentaire.
	"""
	
	cnt_tab = []
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

	return cnt_tab

def tokenize(stringtab):
	"""MIEUX FAIT PAR NLTK
	Sépare chaque string du tableau en un tableau de ses mots et enlève la ponctuation.

	ENTREE
	-----------
	stringtab : array de strings
		Le tableau de commentaires

	SORTIE
	-----------
	tokenized_tab : array d'arrays de strings
		Le tableau des commentaires tokenizé
	"""

	tok_tab = []
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
		
	cnt_tab : array de ints ou de floats
		Tableau contenant pour chaque commentaire le nombre de lettres en majuscule, divisé ou non, selon la valeur de ratio,
		par la longueur du commentaire.
	"""

	cnt_tab = []
	tokenized_tab_low = []
	for i,tokenized_comment in enumerate(tokenized_tab):
		cnt_cap = 0
		cnt = 0
		tab = []
		for word in tokenized_comment:
			for c in word:
				if c == c.upper():
					cnt_cap += 1

				cnt += 1

			tab.append(word.lower())

		tokenized_tab_low.append(tab)

		if ratio:
			cnt_tab.append(cnt_cap/cnt)
		else:
			cnt_tab.append(cnt_cap)

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

def get_bw_stats():
	pass
from nltk.tokenize import TweetTokenizer
from emoji import demojize
import re

tokenizer = TweetTokenizer()

def normalizeToken(token):
	lowercased_token = token.lower()
	if token.startswith("@"):
		return "@USER"
	elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
		return "HTTPURL"
	elif len(token) == 1:
		return demojize(token)
	else:
		if token == "’":
			return "'"
		elif token == "…":
			return "..."
		else:
			return token

def normalizeTweet(tweet):
	tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
	normTweet = " ".join([normalizeToken(token) for token in tokens])

	normTweet = normTweet.replace("cannot ", "can not ").replace("n't ", " n't ").replace("n 't ", " n't ").replace("ca n't", "can't").replace("ai n't", "ain't")
	normTweet = normTweet.replace("'m ", " 'm ").replace("'re ", " 're ").replace("'s ", " 's ").replace("'ll ", " 'll ").replace("'d ", " 'd ").replace("'ve ", " 've ")
	normTweet = normTweet.replace(" p . m .", "  p.m.") .replace(" p . m ", " p.m ").replace(" a . m .", " a.m.").replace(" a . m ", " a.m ")

	normTweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet)
	normTweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet)
	normTweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet)
	
	return " ".join(normTweet.split())
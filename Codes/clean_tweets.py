import re
import emoji
import string
import itertools
import contractions
from nltk.tokenize import word_tokenize


# https://en.wikipedia.org/wiki/Unicode_block
emoji_pattern = re.compile(
	"["
	"\U0001F1E0-\U0001F1FF"  # flags (iOS)
	"\U0001F300-\U0001F5FF"  # symbols & pictographs
	"\U0001F600-\U0001F64F"  # emoticons
	"\U0001F680-\U0001F6FF"  # transport & map symbols
	"\U0001F700-\U0001F77F"  # alchemical symbols
	"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
	"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
	"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
	"\U0001FA00-\U0001FA6F"  # Chess Symbols
	"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
	"\U00002702-\U000027B0"  # Dingbats
	"\U000024C2-\U0001F251" 
	"]+", flags=re.UNICODE)

# emoticons source: https://en.wikipedia.org/wiki/List_of_emoticons
emoticons = {
	":‑)":"smiley",
	":-]":"smiley",
	":-3":"smiley",
	":->":"smiley",
	"8-)":"smiley",
	":-}":"smiley",
	":)":"smiley",
	":]":"smiley",
	":3":"smiley",
	":>":"smiley",
	"8)":"smiley",
	":}":"smiley",
	":o)":"smiley",
	":c)":"smiley",
	":^)":"smiley",
	"=]":"smiley",
	"=)":"smiley",
	":-))":"smiley",
	":‑D":"smiley",
	"8‑D":"smiley",
	"x‑D":"smiley",
	"X‑D":"smiley",
	":D":"smiley",
	"8D":"smiley",
	"xD":"smiley",
	"XD":"smiley",
	":‑(":"sad",
	":‑c":"sad",
	":‑<":"sad",
	":‑[":"sad",
	":(":"sad",
	":c":"sad",
	":<":"sad",
	":[":"sad",
	":-||":"sad",
	">:[":"sad",
	":{":"sad",
	":@":"sad",
	">:(":"sad",
	":'‑(":"sad",
	":'(":"sad",
	":‑P":"playful",
	"X‑P":"playful",
	"x‑p":"playful",
	":‑p":"playful",
	":‑Þ":"playful",
	":‑þ":"playful",
	":‑b":"playful",
	":P":"playful",
	"XP":"playful",
	"xp":"playful",
	":p":"playful",
	":Þ":"playful",
	":þ":"playful",
	":b":"playful",
	"<3":"love"
}


# Read file containing self defined contractions and their expanded forms/phrases
#CONTRACTIONS source: https://en.wikipedia.org/wiki/Contraction_%28grammar%29
self_contractions = {}
with open("contractions.txt", "r") as f_in:
	for line in f_in.readlines():
		kv = line.strip().split(':')
		self_contractions[kv[0]] = kv[1]


# Read file containing slangs and their expanded forms/phrases
slangs = {}
with open("slang.txt", "r") as myCSVfile:
	# Reading file as CSV with delimiter as "=", so that abbreviations are stored in row[0] and phrases in row[1]
	# dataFromFile = csv.reader(myCSVfile, delimiter="=")
	for line in myCSVfile.readlines():
		kv = line.strip().split('=')
		slangs[kv[0]] = kv[1]


# Function to translate abbreviated slang into its corresponding expanded phrase
def translator(text):	
	words = text.strip().split()
	j = 0
	for word in words:
		# Removing special characters...
		word = re.sub('[^a-zA-Z0-9-_.]', '', word)
		# for row in dataFromFile:
		# 	# Check if CAPITAL form of the selected word matches any LHS in the text file.
		# 	if word.upper() == row[0]:
		# 		# If match found, replace it with its expanded form from the text file.
		# 		words[j] = row[1]	
		if word.upper() in slangs:
			words[j] = slangs[word.upper()]
		j = j + 1
	
	return ' '.join(words).strip()


# Function to clean tweets
def clean_tweet(tweet):
	# print("\nOriginal tweet:", tweet)

	text = tweet.strip()

	# Remove extra whitespaces (including new line characters)
	text = re.sub(r'\s\s+', r' ', text)
	# text = re.sub(r'[ ]{2, }', r' ', text) # Doesn't seem to work..Above line does the job..

	# Handle apostrophe
	text = text.replace('\x92',"'")

	# Remove mentions and hyperlinks
	text = ' '.join(word for word in text.split() if not (word.startswith('@') or word.startswith('https') or word.startswith('http') or word.startswith('www.')))
	
	# Remove hashtag while keeping the hashtag text
	text = ' '.join(re.sub(r'#', r' ', text).split())

	# Remove RT - retweets
	text=  ' '.join(re.sub(r'(RT|rt)[ ]*@[ ]*[\S]*', r' ', text).split())
	text=  ' '.join(re.sub(r'(RT|rt)[ ]?@', r' ', text).split())

	# Remove mentions(@)
	text = ' '.join(re.sub(r'@[A-Za-z0-9]+', r' ',text).split())
	text = ' '.join(re.sub(r'@[\S]+', r' ', text).split())	
	
	# Remove URLs and hyperlinks
	text = ' '.join(re.sub(r'\w+:\/\/[\S]+', r' ', text).split())
	text = ' '.join(re.sub(r'(http|https):\/\/[\S]*|www\.[\S]*', r' ', text).split())	

	# Handle &, < and >
	text = re.sub(r'&amp;', r' and ', text)
	text = re.sub(r' \& ', r' and ', text)
	text = re.sub(r'&lt;', r' < ', text)
	text = re.sub(r'&gt;', r' > ', text)

	# Replace emoticons with their sentiment phrases
	words = text.split()
	reformed = [' ' + emoticons[word] + ' ' if word in emoticons else word for word in words]
	text = ' '.join(reformed)

	# Demojize emojis - Expands the emojis into phrases
	text = emoji.demojize(text)
	text = emoji_pattern.sub(r' ', text)

	# # Replace consecutive non-ascii characters with a space
	# text = re.sub(r'[^\x00-\x7F]+', r' ', text)

	# Remove emojis and non-ascii characters	
	text = text.encode('ascii', 'ignore').decode('ascii')
	
	# # Remove non-ascii words and characters
	# text = ''.join(ch if ord(ch) < 128 else '' for ch in text)
	# text = re.sub(r'_[\S]?', r'', text)

	# Cleaning UTF-8 BOM (Byte Order Mark)
	try:
		text = text.decode("utf-8-sig").replace(u"\ufffd", "?")
	except:
		text = text

	# # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
	# text= ''.join(ch for ch in text if ch <= '\uFFFF')

	# # Remove Mojibake (also extra spaces)
	# text = ' '.join(re.sub("[^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a]", " ", text).split())
	
	# Expand the contractions
	text = text.replace("’","'")
	text = contractions.fix(text)
	words = text.split()
	reformed = [self_contractions[word.lower()] if word.lower() in self_contractions else word for word in words]
	text = ' '.join(reformed)

	# Expand slangs
	text = translator(text)

	# Remove misspelling words
	text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))	

	# Remove HTML special entities (e.g. &amp;)
	# Though &amp is replaced with 'and' above..Check for other HTML entities..
	text = re.sub(r'\&[\S]*;', r' ', text)

	# Removing HTML tags
	text = re.sub(r'<.*?>', r' ', text)

	# Remove tickers
	text = re.sub(r'\$[\S]*', r' ', text)

	# Remove extra whitespaces
	text = ' '.join(text.split()).strip()
	
	# Remove punctuations
	# full_punctuation_list = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]"
	# table = str.maketrans('', '', string.punctuation)
	# words = text.split()
	# stripped = [w.translate(table) for w in words]
	# text = ' '.join(word for word in stripped)	
	# text = ' '.join(re.sub(r'[\.\,\!\?\:\;\-\=]', r' ', text).split())
	text = ''.join([' ' if ch in string.punctuation else ch for ch in text])

	# Remove extra whitespaces (including new line characters)
	text = re.sub(r'\s\s+', r' ', text)
	
	# Remove words with 4 or fewer letters
	# text = re.sub(r'\b\w{1,4}\b', '', text)

	# We do not remove NLTK stopwords as all the negative contractions will be removed 
	# which play a significant role in sentiment analysis.
	# Finally, we remove numbers and consider words only
	# text = ' '.join(re.sub("[^a-zA-Z]", " ", text).split()).strip()
	word_tokens = word_tokenize(text.strip())
	text = ' '.join(word_tokens).strip()

	
	# In case filtered text is blank
	if text == "":
		text = tweet.strip()
		
		# Remove extra whitespaces (including new line characters)
		text = re.sub(r'\s\s+', r' ', text)
		
		# Handle apostrophe
		text = text.replace('\x92',"'")
		text = text.replace("’","'")
		
		# Remove hyperlinks
		text = ' '.join(word for word in text.split() if not (word.startswith('https') or word.startswith('http') or word.startswith('www.')))

		# Remove URLs and hyperlinks
		text = ' '.join(re.sub(r'\w+:\/\/[\S]+', r' ', text).split())
		text = ' '.join(re.sub(r'(http|https):\/\/[\S]*|www\.[\S]*', r' ', text).split())
		
		# Remove hashtag while keeping the hashtag text
		text = ' '.join(re.sub(r'#', r' ', text).split())

		# Remove mention symbol while keeping the mention text
		text = ' '.join(re.sub(r'@', r' ', text).split())

		word_tokens = word_tokenize(text.strip())
		text = ' '.join(word for word in word_tokens if word not in string.punctuation).strip()		
	
	# print("Cleaned Tweet: ", text)
	return text
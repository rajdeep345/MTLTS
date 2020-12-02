import re
import os
import json
import codecs
import torch
import argparse
from clean_tweets import clean_tweet
from TweetNormalizer import normalizeTweet

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

from transformers import BertTokenizer, RobertaTokenizer, RobertaConfig, RobertaModel

# model = "BERT"
# model = "ROBERTA"
model = "BERTWEET"

datapath = 'data/pheme-rnr-dataset/'

if model == "BERT":
	feature_path = 'data/features/tweet-features_bert_tokens.txt'
	attention_path = 'data/features/tweet-features_bert_attention.txt'
	tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

elif model == "ROBERTA":
	feature_path = 'data/features/tweet-features_roberta_tokens.txt'
	attention_path = 'data/features/tweet-features_roberta_attention.txt'
	tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

elif model == "BERTWEET":
	feature_path = 'data/features/tweet-features_bertweet_tokens.txt'
	attention_path = 'data/features/tweet-features_bertweet_attention.txt'
	# Load BPE encoder 
	parser = argparse.ArgumentParser()
	parser.add_argument('--bpe-codes', 
		default="BERTweet_base_transformers/bpe.codes",
		required=False,
		type=str,  
		help='path to fastBPE BPE'
	)
	args = parser.parse_args()
	bpe = fastBPE(args)

	# Load the dictionary  
	vocab = Dictionary()
	vocab.add_from_file("BERTweet_base_transformers/dict.txt")


def encode_plus(text, 
	add_special_tokens=True, 
	max_length=32, 
	pad_to_max_length=True, 
	return_attention_mask=True):

	words = bpe.encode(text)

	if pad_to_max_length:
		if len(words.split()) > (max_length - 2):
			words = ' '.join(words.split()[:(max_length - 2)])

	if add_special_tokens:
		subwords = '<s> ' + words + ' </s>'
	else:
		subwords = words

	input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
	tokens_len = len(input_ids)

	if pad_to_max_length:
		pad_len = max_length - tokens_len
		padding = [1] * pad_len
		input_ids.extend(padding)

	if return_attention_mask:
		pad_len = max_length - tokens_len
		attention_mask = [1] * tokens_len
		attention_mask.extend([0] * pad_len)

	return {'input_ids': input_ids, 'attention_mask': attention_mask}


out_features = open(feature_path, 'w')
out_attention = open(attention_path, 'w')
events = ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting', 'sydneysiege']
data_all = []
for event in events:
	data_event = []
	print(f'\n\nProcessing {event}:\n')
	for r in ['/rumours', '/non-rumours']:
		tweets = os.listdir(datapath + event + r)
		for tweet in tweets:
			for x in ['/source-tweet', '/reactions']:
				tweet_path = datapath + event + r + '/' + tweet + x
				files = os.listdir(tweet_path)
				for file in files:
					file = tweet_path + '/' + file
					with open(file, 'r') as f:
						data = json.load(f)
						flag = 0
						tweet_id = data["id"]
						if x == '/source-tweet':
							source_tweet_id = tweet_id
						reply_to_tweet_id = data["in_reply_to_status_id"]						
						if x == '/reactions':
							if tweet_id == source_tweet_id:
								print(f'{event}{r}/{tweet_id} - tweet_id = source_tweet_id')
								flag = 1
							elif reply_to_tweet_id:
								if reply_to_tweet_id == tweet_id:
									print(f'{event}{r}/{source_tweet_id}/{tweet_id} - tweet_id = reply_to_tweet_id')
									flag = 1
							else:
								print(f'{event}{r}/{source_tweet_id}/{tweet_id} - reply_to_tweet_id:null')
								flag = 1
						if flag == 1:
							continue
						
						orig_tweet = data["text"].strip()
						orig_tweet = orig_tweet.replace("’","'")
						orig_tweet = orig_tweet.replace("\n"," ")
						orig_tweet = re.sub(r'\s\s+', r' ', orig_tweet)
						orig_tweet = ' '.join(orig_tweet.split()).strip()
						
						if model == 'BERTWEET':
							cleaned_tweet = normalizeTweet(orig_tweet)
							encoded_dict = encode_plus(
										cleaned_tweet,
										add_special_tokens=True, 
										max_length=32, 
										pad_to_max_length=True, 
										return_attention_mask=True
									)
						else:
							cleaned_tweet = clean_tweet(orig_tweet)
							encoded_dict = tokenizer.encode_plus(
										cleaned_tweet,				# Sentence to encode.
										add_special_tokens=True,	# Add '[CLS]' and '[SEP]'
										padding='max_length',
										truncation= True,
										max_length=32,				# Pad & truncate all sentences.
										return_attention_mask=True,	# Construct attn. masks.
										# return_tensors='pt'		# Return pytorch tensors.
									)
						
						sr_flag = 'S' if x == '/source-tweet' else 'R'
						
						print(f"{str(tweet_id)}\t{event}\t{sr_flag}\t{str(encoded_dict['input_ids'])}", file=out_features)
						print(f"{str(tweet_id)}\t{event}\t{sr_flag}\t{str(encoded_dict['attention_mask'])}", file=out_attention)
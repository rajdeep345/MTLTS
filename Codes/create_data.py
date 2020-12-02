import re
import os
import json
import string
import codecs
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timezone
from clean_tweets import clean_tweet
from TweetNormalizer import normalizeTweet
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

print(f'\nRetrieving Situational labels..')
tweet_situational = {}
with open('data/features/source_tweets_sit_tagged.txt', 'r') as f_in:
	for line in f_in.readlines():
		line = line.strip()
		if line == '' or line.startswith('Event'):
			continue
		values = line.split('\t')
		event = values[0]
		tweet_id = values[1]
		situational = 1 if str(values[-3].strip()) == '1' else 0
		_id = event + ':' + str(tweet_id)
		if _id in tweet_situational:
			print(f'Issue with tweet_id: {_id}')
			continue
		else:
			tweet_situational[_id] = situational		

print(f'\nRetrieving Ground Truth Summary labels..')
tweet_gt_summ = {}
datasets = os.listdir('data/gt_summ/')
for dataset in datasets:
	event = dataset.split('.')[0].strip()
	with open('data/gt_summ/' + dataset, 'r') as f_in:
		for line in f_in.readlines():
			line = line.strip()
			values = line.split('\t')
			tweet_id = values[0]
			_id = event + ':' + str(tweet_id)
			tweet_gt_summ[_id] = 1

print(f'\nRetrieving Veracity labels..')
tweet_veracity = {}
with open('data/features/all_tweets_v2.txt') as f_in:
	for line in f_in.readlines():
		line = line.strip()
		if line == '' or line.startswith('DateTime'):
			continue
		values = line.split('\t')
		event = values[-1]
		tweet_id = values[1]
		sr_flag = values[3]
		rnr_flag = values[4]
		veracity = values[5]
		_id = event + ':' + str(tweet_id)
		if _id in tweet_veracity:
			# print(f'Issue with tweet_id: {_id}')
			continue
		else:
			if sr_flag == 'S':
				if rnr_flag == '1':
					tweet_veracity[_id] = int(veracity)
				else:
					assert veracity == 'None'
					tweet_veracity[_id] = 3
			else:
				tweet_veracity[_id] = 4

print(f'\nRetrieving Stance labels..')
tweet_stance = json.load(open('data/rumoureval2019/train-key.json'))['subtaskaenglish']
tweet_stance.update(json.load(open('data/rumoureval2019/dev-key.json'))['subtaskaenglish'])
all_values = list(set(tweet_stance.values()))
all_values.append('none')
le = LabelEncoder()
le.fit(all_values)

def get_emb(x):
	return model.encode([x['Clean_Tweet']])

def get_norm_emb(x):
	return model.encode([x['Norm_Tweet']])

datapath = 'data/pheme-rnr-dataset/'
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
								# print(f'{event}{r}:{tweet_id} - tweet_id = source_tweet_id')
								flag = 1
							elif reply_to_tweet_id:
								if reply_to_tweet_id == tweet_id:
									# print(f'{event}{r}:{source_tweet_id}-{tweet_id} - tweet_id = reply_to_tweet_id')
									flag = 1
							else:
								# print(f'{event}{r}:{source_tweet_id}-{tweet_id} - reply_to_tweet_id:null')
								flag = 1
						if flag == 1:
							continue
						user = data["user"]["id"]
						date = datetime.strptime(data["created_at"], '%a %b %d %H:%M:%S %z %Y')
						orig_tweet = data["text"].strip()
						orig_tweet = orig_tweet.replace("’","'")
						orig_tweet = orig_tweet.replace("\n"," ")
						orig_tweet = re.sub(r'\s\s+', r' ', orig_tweet)
						orig_tweet = ' '.join(orig_tweet.split()).strip()
						cleaned_tweet = clean_tweet(orig_tweet)
						norm_tweet = normalizeTweet(orig_tweet)
						sr_flag = 'S' if x == '/source-tweet' else 'R'
						rumour_flag = 1 if r == '/rumours' else 0
						search_id = event + ':' + str(tweet_id)
						veracity = tweet_veracity[search_id]
						isSituational = tweet_situational[search_id] if search_id in tweet_situational else 2
						summ_gt = 1 if search_id in tweet_gt_summ else 0
						stance = 'none' if str(tweet_id) not in tweet_stance else tweet_stance[str(tweet_id)]
						entry = [event, str(tweet_id), user, date, orig_tweet, cleaned_tweet, norm_tweet, sr_flag, rumour_flag, veracity, isSituational, summ_gt, stance]
						data_event.append(entry)
						data_all.append(entry)

	df = pd.DataFrame(data_event, columns=['Event', 'Tweet_ID', 'User_ID', 'Date', 'Orig_Tweet', 'Clean_Tweet', 'Norm_Tweet', 'SR', 'R1NR0', 'False0_True1_Unveri2_NR3_Rep4', 'Situ1_NonSitu0_Oth2', 'Summary_gt', 'Stance'])
	df['StanceLabel'] = le.transform(df.Stance)
	# df['Clean_Emb'] = list(model.encode(list(df['Clean_Tweet'].values)))
	# df['Norm_Emb'] = list(model.encode(list(df['Norm_Tweet'].values)))
	df['Clean_Emb'] = df.apply(lambda x: get_emb(x), axis=1)
	df['Norm_Emb'] = df.apply(lambda x: get_norm_emb(x), axis=1)
	total_tweets = len(df)
	total_source_tweets = len(df[df.SR == 'S'])
	total_reply_tweets = total_tweets - total_source_tweets
	assert len(df[df.False0_True1_Unveri2_NR3_Rep4 == 4]) == total_reply_tweets
	assert len(df[df.Situ1_NonSitu0_Oth2 == 2]) == total_reply_tweets
	print(f'\nTotal tweets:{total_tweets}')
	print(f'Total source tweets:{total_source_tweets}')
	print(f'Total reply tweets:{total_reply_tweets}')
	print(f'Total ground truth summary tweets:{len(df[df.Summary_gt == 1])}')
	df.to_pickle('data/features/' + event + '_data.pkl')
	df.to_csv('data/features/' + event + '_data.tsv', sep='\t', index=False)
	
	# df = df[df.SR == 'S'].reset_index(drop=True)
	# df.drop(['Clean_Tweet', 'Normalized_Tweet', 'Tweet_Type_SR', 'False0_True1_Unveri2', 'Stance', 'StanceLabel'], axis=1, inplace=True)
	# df = df[['Event', 'Tweet_ID', 'User_ID', 'Date', 'Orig_Tweet', 'R1NR0']]	
	# df.to_csv('data/features/' + event + '_data_forSitu.tsv', sep='\t', index=False)
	
	print(f'\n\n{event} done..\n\n')

df = pd.DataFrame(data_all, columns=['Event', 'Tweet_ID', 'User_ID', 'Date', 'Orig_Tweet', 'Clean_Tweet', 'Norm_Tweet', 'SR', 'R1NR0', 'False0_True1_Unveri2_NR3_Rep4', 'Situ1_NonSitu0_Oth2', 'Summary_gt', 'Stance'])
df['StanceLabel'] = le.transform(df.Stance)
total_tweets = len(df)
total_source_tweets = len(df[df.SR == 'S'])
total_reply_tweets = total_tweets - total_source_tweets
assert len(df[df.False0_True1_Unveri2_NR3_Rep4 == 4]) == total_reply_tweets
assert len(df[df.Situ1_NonSitu0_Oth2 == 2]) == total_reply_tweets
print(f'\nTotal tweets overall:{total_tweets}')
print(f'\nTotal source tweets overall:{total_source_tweets}')
print(f'Total reply tweets overall:{total_reply_tweets}')
print(f'Total ground truth summary tweets overall:{len(df[df.Summary_gt == 1])}')
df.to_pickle('data/features/all_data.pkl')
df.to_csv('data/features/all_data.tsv', sep='\t', index=False)

# df.drop(['Clean_Tweet', 'Normalized_Tweet', 'Tweet_Type_SR', 'False0_True1_Unverified2', 'Stance', 'StanceLabel'], axis=1, inplace=True)
# df = df[['Event', 'Tweet_ID', 'User_ID', 'Date', 'Orig_Tweet', 'R1NR0']]
# df.to_csv('data/features/all_data_forSitu.tsv', sep='\t', index=False)

import os
import json
import copy
import codecs
import pickle
import operator
import numpy as np
import pandas as pd
import time
from datetime import datetime
from datetime import timezone
from collections import defaultdict

# model = 'BERT'
# model = 'ROBERTA'
model = 'BERTWEET'

datapath = "data/pheme-rnr-dataset/"

if model == "BERT":
	feature_path = "data/features/tweet-features_bert_tokens.txt"
	attention_path = "data/features/tweet-features_bert_attention.txt"
	output_path = "data/features/PT_PHEME5_FeatBERT40_Depth5_maxR5_MTL_Final/"
	
elif model == "ROBERTA":
	feature_path = "data/features/tweet-features_roberta_tokens.txt"
	attention_path = "data/features/tweet-features_roberta_attention.txt"
	output_path = "data/features/PT_PHEME5_FeatROBERTA40_Depth5_maxR5_MTL_Final/"

elif model == "BERTWEET":
	feature_path = "data/features/tweet-features_bertweet_tokens.txt"
	attention_path = "data/features/tweet-features_bertweet_attention.txt"
	output_path = "data/features/PT_PHEME5_FeatBERTWEET40_Depth5_maxR5_MTL_Final/"

features2_path = "data/features/all_tweets_posteriors.txt"

if not os.path.exists(output_path):
	os.makedirs(output_path)

CUTOFF = 5

class Graph:  
	# Constructor 
	def __init__(self, init_dict):  
		# default dictionary to store graph 
		self.graph = defaultdict(list)
		self.DICT_TREE = init_dict
		self.visited = {}  #[False] * (len(self.graph))
  
	# function to add an edge to graph 
	def addEdge(self, u, v): 
		self.graph[u].append(v) 
		self.visited[u] = False
		self.visited[v] = False
  
	# A function used by DFS 
	def DFSUtil(self, v, level):  
		# Mark the current node as visited  
		# and print it 
		self.visited[v] = True

		if level > CUTOFF:
			return

		# print(v, end = ' ')

		# Recur for all the vertices 
		# adjacent to this vertex 
		for i in self.graph[v]: 
			if self.visited[i] == False: 
				self.DFSUtil(i, level + 1)

				if 'cl' in self.DICT_TREE[i]:
					for elem in self.DICT_TREE[i]['cl']:
						if self.DICT_TREE[elem] == {}:
							continue
						self.DICT_TREE[i]['c'].append(self.DICT_TREE[elem])
				
					del self.DICT_TREE[i]['cl']

		if 'cl' in self.DICT_TREE[v]:
			for elem in self.DICT_TREE[v]['cl']:
				if self.DICT_TREE[elem] == {}:
					continue
				self.DICT_TREE[v]['c'].append(self.DICT_TREE[elem])	
		
			del self.DICT_TREE[v]['cl']

  
	# The function to do DFS traversal. It uses 
	# recursive DFSUtil() 
	def DFS(self, v):  
		# Call the recursive helper function  
		# to print DFS traversal 
		self.DFSUtil(v, 1)


def main():
	FEATURES = {}
	ATTENTION = {}
	feature_file = codecs.open(feature_path, 'r', 'utf-8')
	attention_file = codecs.open(attention_path,'r','utf-8')
	for row in feature_file:
		s = row.strip().split('\t')
		FEATURES[s[0]] = eval(s[-1].strip())
	for row in attention_file:
		s = row.strip().split('\t')
		ATTENTION[s[0]] = eval(s[-1].strip())

	####################### comment out if 40 features not needed ################# 
	OLD_FEATURES = {}
	old_features = codecs.open(features2_path, 'r', 'utf-8')
	for row in old_features:
		s = row.strip().split('\t')
		OLD_FEATURES[(s[1])] = eval(s[-2].strip())
	###########################################################################

	feature_file.close()
	attention_file.close()
	old_features.close()
	print('Features Loaded:', len(FEATURES))
	print('attentions Loaded:', len(ATTENTION))
	print('Old Features Loaded:', len(OLD_FEATURES))

	dfc = pd.read_pickle('data/features/summary_dataframes/dfc_0.57.pkl')
	dfg = pd.read_pickle('data/features/summary_dataframes/dfg_0.72.pkl')
	dfo = pd.read_pickle('data/features/summary_dataframes/dfo_0.6.pkl')
	dfs = pd.read_pickle('data/features/summary_dataframes/dfs_0.6.pkl')

	datasets = os.listdir(datapath)
	for dataset in datasets:
		if dataset == '.DS_Store' or dataset == 'README':
			continue

		print(f'\n\n{dataset} started..')
		rumour_path = datapath + dataset + "/rumours/"
		non_rumour_path = datapath + dataset + "/non-rumours/"

		dict_r = {}
		dict_label = {}
		dict_graph = {}
		LABELS = {}
		reactions = set()
		all_disjoint_reactions = set()
		err_cnt = 0

		# Rumour
		print("\n\nHandling rumours...")
		rumour_tweets = os.listdir(rumour_path)
		for tweet_id in rumour_tweets:
			if tweet_id == '.DS_Store':
				continue			
			tweet_id = str(tweet_id)
			print(f'\nSource TweetID: {tweet_id}')
			source_path = rumour_path + tweet_id + "/source-tweet/" + tweet_id + '.json'
			source_tweet_file = codecs.open(source_path, 'r', 'utf-8')
			tweet = source_tweet_file.read()
			d = json.loads(tweet)
			date_created = datetime.strptime(d["created_at"], '%a %b %d %H:%M:%S %z %Y')#.replace(tzinfo=timezone.utc).astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S'))
			ts = time.mktime(date_created.timetuple())
			source_tweet_file.close()

			if tweet_id not in dict_r:
				# print(str(tweet_id) + " not in dict_r..")
				dict_r[tweet_id] = [(tweet_id, ts)]
				dict_graph[tweet_id] = []
				# print("After inserting..")
				# print("dict_r[" + str(tweet_id) + "] = " + str(dict_r[tweet_id]))
				# print("dict_graph[" + str(tweet_id) + "] = " + str(dict_graph[tweet_id]))		

			reactions_path = rumour_path + tweet_id + "/reactions/"
			reaction_tweets = os.listdir(reactions_path)
			disjoint_reactions = set()			

			# if(len(reaction_tweets)>5):
			# 	reaction_tweets = reaction_tweets[:5]
			reaction_tweet_count = 0

			for r_tweet_id in reaction_tweets:
				if reaction_tweet_count >= CUTOFF:
					break
				if r_tweet_id == '.DS_Store':
					continue
				reaction_tweet_file = codecs.open(reactions_path + r_tweet_id, 'r', 'utf-8')
				r_tweet = reaction_tweet_file.read()
				d = json.loads(r_tweet)
				date_created = datetime.strptime(d["created_at"], '%a %b %d %H:%M:%S %z %Y')
				ts = time.mktime(date_created.timetuple())				
				reaction_tweet_file.close()				
				
				r_id = str(r_tweet_id.split('.')[0])
				source_id = str(d['in_reply_to_status_id'])
				if r_id == str(tweet_id):
					print(f'{dataset}/rumours: Reply TweetID = Source TweetID = {r_id} - Hence ignored...')
					continue
				elif source_id:
					if source_id == r_id:
						print(f'{dataset}/rumours: Reply TweetID = reply_to_tweet_id = {r_id} - Hence ignored...')
						continue					
				else:
					print(f'{dataset}/rumours: Reply TweetID: {r_id} - reply_to_tweet_id:null - Hence ignored...')
					continue
					
				if source_id not in dict_graph:
					print(f'\nParent TweetID: {source_id} not in dict_graph..')
					print(f'{r_id} is a disjoint reply..Ignoring..')
					disjoint_reactions.add(r_tweet_id.split('.')[0])
					all_disjoint_reactions.add(r_tweet_id.split('.')[0])
					continue					
					# dict_graph[source_id] = []
					# print("After inserting..")
					# print("dict_graph[" + str(source_id) + "] = " + str(dict_graph[source_id]))

				dict_r[tweet_id].append((r_id, ts))
				dict_graph[source_id].append(r_id)
				# print("After updating dict_r[tweet_id] and dict_graph[source_id]")
				# print("dict_r[" + str(tweet_id) + "] = " + str(dict_r[tweet_id]))
				# print("dict_graph[" + str(source_id) + "] = " + str(dict_graph[source_id]))

				if r_id not in dict_graph:
					# print("Reply TweetID: " + str(r_id) + " not in dict_graph")
					dict_graph[r_id] = []
					# print("After inserting..")
					# print("dict_graph[" + str(r_tweet_id.split('.')[0]) + "] = " + str(dict_graph[r_tweet_id.split('.')[0]]))

				reaction_tweet_count += 1				
				print(f'Reply TweetID: {r_id}')
				reactions.add(r_tweet_id)
				LABELS[r_id] = [0, 1]

			dict_label[tweet_id] = 'r'
			LABELS[tweet_id] = [0, 1]

		rumor_source_tweet_count = len(dict_label)
		rumor_reply_tweet_count = len(list(reactions))
		rumor_disjoint_reply_count = len(list(all_disjoint_reactions))
		
		
		# Non_Rumour
		print("\n\nHandling non-rumours...")
		non_rumour_tweets = os.listdir(non_rumour_path)
		for tweet_id in non_rumour_tweets:
			if tweet_id == '.DS_Store':
				continue
			tweet_id = str(tweet_id)
			print(f'\nSource TweetID: {tweet_id}')
			source_path = non_rumour_path + tweet_id + "/source-tweet/" + tweet_id + '.json'
			source_tweet_file = codecs.open(source_path, 'r', 'utf-8')
			tweet = source_tweet_file.read()
			d = json.loads(tweet)
			date_created = datetime.strptime(d["created_at"], '%a %b %d %H:%M:%S %z %Y')
			ts = time.mktime(date_created.timetuple())
			source_tweet_file.close()

			if tweet_id not in dict_r:
				# print(str(tweet_id) + " not in dict_r..")
				dict_r[tweet_id] = [(tweet_id, ts)]
				dict_graph[tweet_id] = []
				# print("After inserting..")
				# print("dict_r[" + str(tweet_id) + "] = " + str(dict_r[tweet_id]))
				# print("dict_graph[" + str(tweet_id) + "] = " + str(dict_graph[tweet_id]))

			reactions_path = non_rumour_path + tweet_id + "/reactions/"
			reaction_tweets = os.listdir(reactions_path)
			disjoint_reactions = set()
			
			# if(len(reaction_tweets)>5):
			# 	reaction_tweets = reaction_tweets[:5]
			reaction_tweet_count = 0

			for r_tweet_id in reaction_tweets:
				if reaction_tweet_count >= CUTOFF:
					break
				if r_tweet_id == '.DS_Store':
					continue
				reaction_tweet_file = codecs.open(reactions_path + r_tweet_id, 'r', 'utf-8')
				r_tweet = reaction_tweet_file.read()
				d = json.loads(r_tweet)
				date_created = datetime.strptime(d["created_at"], '%a %b %d %H:%M:%S %z %Y')
				ts = time.mktime(date_created.timetuple())				
				reaction_tweet_file.close()				
				
				r_id = str(r_tweet_id.split('.')[0])
				source_id = str(d['in_reply_to_status_id'])
				if r_id == str(tweet_id):
					print(f'{dataset}/non-rumours: Reply TweetID = Source TweetID = {r_id} - Hence ignored...')
					continue
				elif source_id:
					if source_id == r_id:
						print(f'{dataset}/non-rumours: Reply TweetID = reply_to_tweet_id = {r_id} - Hence ignored...')
						continue					
				else:
					print(f'{dataset}/non-rumours: Reply TweetID: {r_id} - reply_to_tweet_id:null - Hence ignored...')
					continue

				if source_id not in dict_graph:
					print(f'\nParent TweetID: {source_id} not in dict_graph..')
					print(f'{r_id} is a disjoint reply..Ignoring..')
					disjoint_reactions.add(r_tweet_id.split('.')[0])
					all_disjoint_reactions.add(r_tweet_id.split('.')[0])
					continue					
					# dict_graph[source_id] = []
					# print("After inserting..")
					# print("dict_graph[" + str(source_id) + "] = " + str(dict_graph[source_id]))

				dict_r[tweet_id].append((r_id, ts))
				dict_graph[source_id].append(r_id)
				# print("After updating dict_r[tweet_id] and dict_graph[source_id]")
				# print("dict_r[" + str(tweet_id) + "] = " + str(dict_r[tweet_id]))
				# print("dict_graph[" + str(source_id) + "] = " + str(dict_graph[source_id]))

				if r_id not in dict_graph:
					# print("Reply TweetID: " + str(r_id) + " not in dict_graph")
					dict_graph[r_id] = []
					# print("After inserting..")
					# print("dict_graph[" + str(r_tweet_id.split('.')[0]) + "] = " + str(dict_graph[r_tweet_id.split('.')[0]]))

				reaction_tweet_count += 1				
				print(f'Reply TweetID: {r_id}')
				reactions.add(r_tweet_id)
				LABELS[r_id] = [1, 0]

			dict_label[tweet_id] = 'nr'
			LABELS[tweet_id] = [1, 0]

		non_rumor_source_tweet_count = len(dict_label) - rumor_source_tweet_count
		non_rumor_reply_tweet_count = len(list(reactions)) - rumor_reply_tweet_count
		non_rumor_disjoint_reply_count = len(list(all_disjoint_reactions)) - rumor_disjoint_reply_count

		print('\n\nTotal no. of rumor source tweets: ', rumor_source_tweet_count)
		print('Total no. of rumor reply tweets:', rumor_reply_tweet_count)
		print('Total no. of rumor disjoint reply tweets:', rumor_disjoint_reply_count)
		print('Total no. of non-rumor source tweets: ', non_rumor_source_tweet_count)
		print('Total no. of non-rumor reply tweets:', non_rumor_reply_tweet_count)
		print('Total no. of non-rumor disjoint reply tweets:', non_rumor_disjoint_reply_count)


		if dataset == 'charliehebdo':
			temp_df = dfc.copy(deep=False)
		elif dataset == 'germanwings-crash':
			temp_df = dfg.copy(deep=False)
		elif dataset == 'ottawashooting':
			temp_df = dfo.copy(deep=False)
		else:
			temp_df = dfs.copy(deep=False)

		curr_tree = {}
		IDs = set()
		count = 0
		output_file = codecs.open(output_path + "/" + dataset + '.txt', 'w', 'utf-8')
		in_summ_orig = 0
		in_summ_clean = 0
		in_summ_bt = 0
		for tweet_id in dict_graph:
			# if tweet_id not in dict_r:
			# 	continue			
			try:
				curr_tree[tweet_id] = {}
				#############################################################
				################### Only for source tweets ##################
				# if int(tweet_id) in temp_df['Tweet_ID'].values:
				# 	count += 1
				# 	orig_tweet_text = temp_df.loc[temp_df.Tweet_ID==int(tweet_id), 'Orig_Tweet'].values[0]
				# 	curr_tree[tweet_id]['orig_tweet'] = orig_tweet_text
				# 	clean_tweet_text = temp_df.loc[temp_df.Tweet_ID==int(tweet_id), 'Clean_Tweet'].values[0]
				# 	curr_tree[tweet_id]['clean_tweet'] = clean_tweet_text
				#############################################################
				curr_tree[tweet_id]['f'] = FEATURES[tweet_id]
				curr_tree[tweet_id]['a'] = ATTENTION[tweet_id]
				########## comment if 40 features not required ##############
				curr_tree[tweet_id]['k'] = OLD_FEATURES[tweet_id]
				#############################################################
				curr_tree[tweet_id]['l'] = LABELS[tweet_id]
				########## Additional labels only for source tweets #########
				if dataset != 'ferguson':
					if tweet_id in temp_df['Tweet_ID'].values:
						count += 1
						# sit_label = temp_df.loc[temp_df.Tweet_ID==tweet_id, 'Situ1_NonSitu0_Oth2'].values[0]
						# curr_tree[tweet_id]['sit_gt'] = sit_label
						orig_summ_label = temp_df.loc[temp_df.Tweet_ID==tweet_id, 'Summary_gt'].values[0]
						curr_tree[tweet_id]['orig_summ_gt'] = orig_summ_label
						if orig_summ_label == 1:
							in_summ_orig += 1
						new_summ_label = temp_df.loc[temp_df.Tweet_ID==tweet_id, 'New_Summ_gt_Clean'].values[0]
						curr_tree[tweet_id]['summ_gt_clean'] = new_summ_label
						if new_summ_label == 1:
							in_summ_clean += 1
						new_summ_label = temp_df.loc[temp_df.Tweet_ID==tweet_id, 'New_Summ_gt_BT'].values[0]
						curr_tree[tweet_id]['summ_gt_bt'] = new_summ_label
						if new_summ_label == 1:
							in_summ_bt += 1
						# class_label = temp_df.loc[temp_df.Tweet_ID==int(tweet_id), 'Class'].values[0]
						# curr_tree[tweet_id]['class_gt'] = class_label
				#############################################################
				curr_tree[tweet_id]['cl'] = copy.copy(dict_graph[tweet_id])
				curr_tree[tweet_id]['c'] = []
				IDs.add(tweet_id)
			except:
				print('Error:', tweet_id)
				err_cnt += 1

		print(f'\nTotal source tweets for {dataset}: {count}')
		print(f'In-Summary: Original:{in_summ_orig}, Clean:{in_summ_clean}, Bertweet:{in_summ_bt}')
	
		g = Graph(curr_tree)

		for tweet_id in IDs:
			curr_li = dict_graph[tweet_id]
			for curr_elem in curr_li:
				if curr_elem in IDs:
					g.addEdge(tweet_id, curr_elem)		
		
		for tweet_id in dict_label:
			if tweet_id not in dict_r or tweet_id not in IDs:
				continue
			# print(tweet_id)			
			g.DFS(tweet_id)
			# exit(-1)
			# output_file.write(tweet_id + '\t' + str(g.DICT_TREE[tweet_id]))
			print(tweet_id + '\t' + str(g.DICT_TREE[tweet_id]), file=output_file)

		output_file.close()

		print(dataset + ' done with Error Count: ' + str(err_cnt) + '\n')


if __name__ == "__main__":main()

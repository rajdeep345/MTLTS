import os
import sys
import math
import string
import argparse
import codecs
import random
import numpy
import pandas as pd
import itertools
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import Dataset, IterableDataset, DataLoader, TensorDataset
from torch.utils.data import random_split, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import *
# import matplotlib.pyplot as plt
# from tqdm import tqdm

import re
import copy
import time
import datetime
import rouge
import textstat
import subprocess
import logging
logging.basicConfig(level=logging.ERROR)

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu_id', type=int, default=0)
	parser.add_argument('--model_name', type=str, default="BERT") 
	parser.add_argument('--in_features', type=int, default=768)
	parser.add_argument('--save_policy', type=str, default="loss")
	parser.add_argument('--veri_loss_fn', type=str, default="nw")
	parser.add_argument('--summ_loss_fn', type=str, default="w")
	parser.add_argument('--optim', type=str, default="adam")
	parser.add_argument('--l2', type=str, default="n")
	parser.add_argument('--wd', type=float, default=0.01)
	parser.add_argument('--use_dropout', type=str, default="n")
	parser.add_argument('--classifier_dropout', type=float, default=0.2)
	parser.add_argument('--iters', type=int, default=5)
	parser.add_argument('--bs', type=int, default=16)
	parser.add_argument('--seed', type=int, default=1955)
	parser.add_argument('--lambda1', type=float, default=0.5)
	parser.add_argument('--lambda2', type=float, default=0.5)
	parser.add_argument('--flood', type=str, default="n")
	parser.add_argument('--test_file', type=str, default="german")
	parser.add_argument('--lr', type=float, default=1e-5)

	args = parser.parse_args()

	gpu_id = args.gpu_id
	print(f'GPU_ID = {gpu_id}\n')   
	# MODEL_NAME = args.model_name
	MODEL_NAME = 'BERT'
	print(f'MODEL_NAME = {MODEL_NAME}')
	IN_FEATURES = args.in_features
	print(f'IN_FEATURES = {IN_FEATURES}')
	OUT_FEATURES = 128
	print(f'OUT_FEATURES = {OUT_FEATURES}')
	MODEL_SAVING_POLICY = args.save_policy
	print(f'MODEL_SAVING_POLICY = {MODEL_SAVING_POLICY}')
	VERI_LOSS_FN = args.veri_loss_fn
	print(f'VERI_LOSS_FN = {VERI_LOSS_FN}')
	SUMM_LOSS_FN = args.summ_loss_fn
	print(f'SUMM_LOSS_FN = {SUMM_LOSS_FN}')
	LAMBDA1 = args.lambda1
	print(f'LAMBDA1 = {LAMBDA1}')
	LAMBDA2 = args.lambda2
	print(f'LAMBDA2 = {LAMBDA2}')
	FLOOD = args.flood
	print(f'FLOOD = {FLOOD}')
	OPTIM = args.optim
	print(f'OPTIM = {OPTIM}')
	L2_REGULARIZER = args.l2
	print(f'L2_REGULARIZER = {L2_REGULARIZER}')
	WEIGHT_DECAY = args.wd
	if L2_REGULARIZER == 'y':
		print(f'WEIGHT_DECAY = {WEIGHT_DECAY}')
	USE_DROPOUT = args.use_dropout
	print(f'USE_DROPOUT = {USE_DROPOUT}')
	CLASSIFIER_DROPOUT = args.classifier_dropout
	if USE_DROPOUT == 'y':
		print(f'CLASSIFIER_DROPOUT = {CLASSIFIER_DROPOUT}')
	NUM_ITERATIONS = args.iters
	print(f'NUM_ITERATIONS = {NUM_ITERATIONS}')
	BATCH_SIZE = args.bs
	print(f'BATCH_SIZE = {BATCH_SIZE}')
	# lr_list = [5e-6, 1e-5, 2e-5, 5e-5, 1e-4]
	lr_list = [args.lr]
	print(f'LEARNING_RATES = {str(lr_list)}')
	TRAINABLE_LAYERS = [0,1,2,3,4,5,6,7,8,9,10,11]
	print(f'TRAINABLE_LAYERS = {str(TRAINABLE_LAYERS)}')
	TEST_FILE = args.test_file
	print(f'\nTEST_FILE = {TEST_FILE}')
	seed_val = args.seed
	print(f'\nSEED = {str(seed_val)}\n\n')


if torch.cuda.is_available():
	if gpu_id == 0:
		device = torch.device("cuda:0")
	else:
		device = torch.device("cuda:1")
	print('There are %d GPU(s) available.' % torch.cuda.device_count())
	print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
	print('No GPU available, using the CPU instead.')
	device = torch.device("cpu")



# To make results reproducable
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)
random.seed(seed_val)
numpy.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)


if MODEL_NAME == 'BERT':	
	tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
elif MODEL_NAME == 'ROBERTA':
	tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

tweetconfig = RobertaConfig.from_pretrained("MTLVS/BERTweet_base_transformers/config.json")

sigmoid_fn = torch.nn.Sigmoid()

class TreeDataset(Dataset):

	def __init__(self, data):
		self.data = data
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		return self.data[idx]


def _label_node_index(node, n=0):
	node['index'] = n
	for child in node['c']:
		n += 1
		_label_node_index(child, n)


def _gather_node_attributes(node, key):
	features = [node[key]]
	for child in node['c']:
		features.extend(_gather_node_attributes(child, key))
	return features


def _gather_adjacency_list(node):
	adjacency_list = []
	for child in node['c']:
		adjacency_list.append([node['index'], child['index']])
		adjacency_list.extend(_gather_adjacency_list(child))

	return adjacency_list


def convert_tree_to_tensors(tree, tweet_id, device=device):
	# Label each node with its walk order to match nodes to feature tensor indexes
	# This modifies the original tree as a side effect
	_label_node_index(tree)

	features = _gather_node_attributes(tree, 'f')
	attention = _gather_node_attributes(tree, 'a')
	old_features = _gather_node_attributes(tree, 'k')
	labels = _gather_node_attributes(tree, 'l')
	tweetid = tweet_id
	root_label = labels[0][1]
	# summ_gt = tree['new_summ_gt']
	summ_gt = tree['summ_gt_clean']
	adjacency_list = _gather_adjacency_list(tree)

	node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(features))
	root_node = [0]

	return {
		'f': torch.tensor(features, dtype=torch.long),
		'a':torch.tensor(attention,  dtype=torch.float32),
		'k':torch.tensor(old_features, dtype=torch.float32),        
		'l': torch.tensor(labels,  dtype=torch.int64),
		'root_l': torch.tensor(root_label, dtype=torch.int64),
		'root_n': torch.tensor(root_node,  dtype=torch.int64),
		'node_order': torch.tensor(node_order,  dtype=torch.int64),
		'adjacency_list': torch.tensor(adjacency_list,  dtype=torch.int64),
		'edge_order': torch.tensor(edge_order,  dtype=torch.int64),
		'summ_gt': torch.tensor(summ_gt, dtype=torch.int64),
		'tweet_id' : torch.tensor(tweet_id, dtype=torch.long)
	}

def calculate_evaluation_orders(adjacency_list, tree_size):
	'''Calculates the node_order and edge_order from a tree adjacency_list and the tree_size.

	The TreeLSTM model requires node_order and edge_order to be passed into the model along
	with the node features and adjacency_list.  We pre-calculate these orders as a speed
	optimization.
	'''
	adjacency_list = numpy.array(adjacency_list)
	node_ids = numpy.arange(tree_size, dtype=int)
	node_order = numpy.zeros(tree_size, dtype=int)
	unevaluated_nodes = numpy.ones(tree_size, dtype=bool)

	if(len(adjacency_list)==0):
		return [0],[]
	parent_nodes = adjacency_list[:, 0]
	child_nodes = adjacency_list[:, 1]

	n = 0
	while unevaluated_nodes.any():
		# Find which child nodes have not been evaluated
		unevaluated_mask = unevaluated_nodes[child_nodes]

		# Find the parent nodes of unevaluated children
		unready_parents = parent_nodes[unevaluated_mask]

		# Mark nodes that have not yet been evaluated
		# and which are not in the list of parents with unevaluated child nodes
		nodes_to_evaluate = unevaluated_nodes & ~numpy.isin(node_ids, unready_parents)

		node_order[nodes_to_evaluate] = n
		unevaluated_nodes[nodes_to_evaluate] = False

		n += 1

	edge_order = node_order[parent_nodes]

	return node_order, edge_order


def batch_tree_input(batch):
	'''Combines a batch of tree dictionaries into a single batched dictionary for use by the TreeLSTM model.

	batch - list of dicts with keys ('f', 'node_order', 'edge_order', 'adjacency_list')
	returns a dict with keys ('f', 'node_order', 'edge_order', 'adjacency_list', 'tree_sizes')
	'''
	tree_sizes = [b['f'].shape[0] for b in batch]

	batched_features = torch.cat([b['f'] for b in batch])
	batched_attentions = torch.cat([b['a'] for b in batch])
	batched_old_features = torch.cat([b['k'] for b in batch])
	batched_node_order = torch.cat([b['node_order'] for b in batch])

	idx = 0
	root_li = []

	for b in batch:
		root_li.append(idx)
		idx += len(b['node_order'])

	batched_root = torch.tensor(root_li, dtype=torch.int64)

	batched_edge_order = torch.cat([b['edge_order'] for b in batch])

	batched_labels = torch.cat([b['l'] for b in batch])

	batched_root_labels = torch.tensor([b['root_l'] for b in batch])
	batched_summ_labels = torch.tensor([b['summ_gt'] for b in batch])
	
	batched_adjacency_list = []
	offset = 0
	for n, b in zip(tree_sizes, batch):
		batched_adjacency_list.append(b['adjacency_list'] + offset)
		offset += n
	batched_adjacency_list = torch.cat(batched_adjacency_list)

	return {
		'f': batched_features,
		'a': batched_attentions,
		'k': batched_old_features,        
		'node_order': batched_node_order,
		'edge_order': batched_edge_order,
		'adjacency_list': batched_adjacency_list,
		'tree_sizes': tree_sizes,
		'root_node': batched_root,
		'root_label': batched_root_labels,
		'l': batched_labels,
		'summ_gt': batched_summ_labels
	}


def unbatch_tree_tensor(tensor, tree_sizes):
	'''Convenience functo to unbatch a batched tree tensor into individual tensors given an array of tree_sizes.
	sum(tree_sizes) must equal the size of tensor's zeroth dimension.
	'''
	return torch.split(tensor, tree_sizes, dim=0)



class TreeLSTM(torch.nn.Module):
	'''PyTorch TreeLSTM model that implements efficient batching.
	'''
	def __init__(self, model_name, trainable_layers, in_features, out_features, classifier_dropout, mode, tweetconfig=tweetconfig):
		'''TreeLSTM class initializer

		Takes in int sizes of in_features and out_features and sets up model Linear network layers.
		'''
		super().__init__()
		print("model intialising...")
		self.in_features = in_features
		self.out_features = out_features
		self.mode = mode
		
		if model_name == 'BERT':
			self.BERT_model = BertModel.from_pretrained("bert-base-cased", output_attentions=True)
		elif model_name == 'ROBERTA':
			self.BERT_model = RobertaModel.from_pretrained("roberta-base", output_attentions=True)
		elif model_name == 'BERTWEET':
			self.BERT_model = RobertaModel.from_pretrained("MTLVS/BERTweet_base_transformers/model.bin", config=tweetconfig)
		
		# for name, param in self.BERT_model.named_parameters():
		#   flag = False
		#   for num in trainable_layers:
		#       if 'layer.'+ str(num) + '.' in name:
		#           param.requires_grad = True
		#           flag = True
		#           break
		#   if not flag:
		#       if 'pooler' in name or 'embedding' in name:
		#           param.requires_grad = True
		#       else:
		#           param.requires_grad = False

		self.W_iou = torch.nn.Linear(self.in_features, 3 * self.out_features)
		self.U_iou = torch.nn.Linear(self.out_features, 3 * self.out_features, bias=False)

		# f terms are maintained seperate from the iou terms because they involve sums over child nodes
		# while the iou terms do not
		self.W_f = torch.nn.Linear(self.in_features, self.out_features)
		self.U_f = torch.nn.Linear(self.out_features, self.out_features, bias=False)
		
		self.fc = torch.nn.Linear(self.out_features, 2)
		
		# self.bert_dropout = torch.nn.Dropout(bert_dropout)		
		self.classifier_dropout = torch.nn.Dropout(classifier_dropout)		
		
		self.summ_fc1 = torch.nn.Linear(self.in_features, self.out_features)		
		self.summ_fc2 = torch.nn.Linear(self.out_features, 2)

		# self.init_weights()


	# def init_weights(self):
	# 	for name, param in self.named_parameters():
	# 		if "BERT" in name or "bias" in name:
	# 			continue
	# 		else:
	# 			torch.nn.init.xavier_uniform_(param)

	
	def forward(self, features, attentions, old_features, node_order, adjacency_list, edge_order, root_node):
		'''Run TreeLSTM model on a tree data structure with node features

		Takes Tensors encoding node features, a tree node adjacency_list, and the order in which 
		the tree processing should proceed in node_order and edge_order.
		'''

		# Total number of nodes in every tree in the batch
		batch_size = node_order.shape[0]

		# Retrive device the model is currently loaded on to generate h, c, and h_sum result buffers
		device = next(self.parameters()).device

		# h and c states for every node in the batch
		# h - hidden state
		# c - memory state
		h = torch.zeros(batch_size, self.out_features, device=device)		
		c = torch.zeros(batch_size, self.out_features, device=device)
		
		bert_outputs = self.BERT_model(input_ids=features, attention_mask=attentions)
		hidden_states = bert_outputs[0]
		att = bert_outputs[-1]

		if self.mode=="cls":
			output_vectors = hidden_states[:,0]
		elif self.mode=="avg":
			input_mask_expanded = attentions.unsqueeze(-1).expand(hidden_states.size()).float()
			sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
			sum_mask = input_mask_expanded.sum(1)
			output_vectors= sum_embeddings / sum_mask
			
		if self.in_features == 808:
			output_vectors = torch.cat([output_vectors, old_features], axis=1)
		
		# output_vectors = self.bert_dropout(output_vectors)

		summ_in = output_vectors[root_node,:]
		summ_logits1 = self.summ_fc1(summ_in)
		summ_logits_out = self.summ_fc2(summ_logits1)
		
		
		for n in range(node_order.max() + 1):
			self._run_lstm(n, h, c, output_vectors, node_order, adjacency_list, edge_order)

		h_root = h[root_node, :]
		
		if USE_DROPOUT == 'y':
			h_root = self.classifier_dropout(h_root)
		
		veri_logits_out = self.fc(h_root)
		
		return h, veri_logits_out, c, summ_logits_out, att 


	def _run_lstm(self, iteration, h, c, features, node_order, adjacency_list, edge_order):
		'''Helper function to evaluate all tree nodes currently able to be evaluated.
		'''
		node_mask = node_order == iteration

		# edge_mask is a tensor of size E x 1
		edge_mask = edge_order == iteration

		x = features[node_mask, :]
		if iteration == 0:
			iou = self.W_iou(x)
		else:
			# adjacency_list is a tensor of size e x 2
			adjacency_list = adjacency_list[edge_mask, :]

			parent_indexes = adjacency_list[:, 0]
			child_indexes = adjacency_list[:, 1]

			# child_h and child_c are tensors of size e x 1
			child_h = h[child_indexes, :]
			child_c = c[child_indexes, :]

			# Add child hidden states to parent offset locations
			_, child_counts = torch.unique_consecutive(parent_indexes, return_counts=True)
			child_counts = tuple(child_counts)
			parent_children = torch.split(child_h, child_counts)
			parent_list = [item.sum(0) for item in parent_children]

			h_sum = torch.stack(parent_list)
			iou = self.W_iou(x) + self.U_iou(h_sum)


		# i, o and u are tensors of size n x M
		i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
		i = torch.sigmoid(i)
		o = torch.sigmoid(o)
		u = torch.tanh(u)

		if iteration == 0:
			c[node_mask, :] = i * u
		else:
			# f is a tensor of size e x M
			f = self.W_f(features[parent_indexes, :]) + self.U_f(child_h)
			f = torch.sigmoid(f)
			# fc is a tensor of size e x M
			fc = f * child_c

			# Add the calculated f values to the parent's memory cell state
			parent_children = torch.split(fc, child_counts)
			parent_list = [item.sum(0) for item in parent_children]

			c_sum = torch.stack(parent_list)
			c[node_mask, :] = i * u + c_sum

		h[node_mask, :] = o * torch.tanh(c[node_mask])


def save_model(model, name, val_acc=0, val_loss=1):
	state = {
		'model':model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'val_acc': val_acc,
		'val_loss': val_loss
		}
	torch.save(state, name)


def load_model(model, name):
	state = torch.load(name)
	model.load_state_dict(state['model'])
	optimizer.load_state_dict(state['optimizer'])
	print('Validation accuracy of the model is ', state.get('val_acc'))
	print('Validation loss of the model is ', state.get('val_loss'))
	return state.get('val_acc')


if MODEL_NAME == 'BERT':
	tree_path = 'MTLVS/data/features/PT_PHEME5_FeatBERT40_Depth5_maxR5_MTL_Final/'
elif MODEL_NAME == 'ROBERTA':
	tree_path = 'MTLVS/data/features/PT_PHEME5_FeatROBERTA40_Depth5_maxR5_MTL_Final/'
elif MODEL_NAME == 'BERTWEET':
	tree_path = 'MTLVS/data/features/PT_PHEME5_FeatBERTWEET40_Depth5_maxR5_MTL_Final/'

files = ['charliehebdo.txt', 'germanwings-crash.txt', 'ottawashooting.txt', 'sydneysiege.txt']


def split_data(trees, frac):
	pos_data = []
	neg_data = []
	for tree in trees:
		# if tree['root_l'].tolist() == [[0, 1]]:
		if tree['summ_gt'].tolist() == [1]:
			pos_data.append(tree)
		else:
			neg_data.append(tree)
	pos_len = int(frac * len(pos_data))
	neg_len = int(frac * len(neg_data))
	val_li = pos_data[:pos_len] + neg_data[:neg_len]
	random.shuffle(val_li)
	train_li = pos_data[pos_len:] + neg_data[neg_len:]
	random.shuffle(train_li)
	return train_li, val_li


tree_li = {}
val_li = {}
for filename in files:
	input_file = codecs.open(tree_path + filename, 'r', 'utf-8')
	tree_li[filename]=[]
	for row in input_file:
		s = row.strip().split('\t')		
		tweet_id = int(s[0])
		curr_tree = eval(s[1])
		curr_tensor = convert_tree_to_tensors(curr_tree, tweet_id)
		tree_li[filename].append(curr_tensor)
		
	random.shuffle(tree_li[filename])
	tree_li[filename], val_li[filename] = split_data(tree_li[filename], 0.1)
	input_file.close()
	print(f'{filename} Training Set Size: {len(tree_li[filename])}, Validation Set Size: {len(val_li[filename])}, Total: {len(tree_li[filename]) + len(val_li[filename])}')

weight_vec = {}
summ_weight_vec = {}
pos_weight_vec = {}
summ_pos_weight_vec = {}
for test_file in files:
	y = []
	summ_y = []
	label_dist = [0, 0]
	summ_label_dist = [0, 0]
	for filename in files:		
		if filename != test_file:			
			file_dist = [0, 0]
			summ_file_dist = [0, 0]
			for tree in tree_li[filename]:
				# veri_label = int(tree['root_l'].tolist()[0][1])
				veri_label = int(tree['root_l'].tolist())
				y.append(veri_label)
				summ_label = int(tree['summ_gt'].tolist())
				summ_y.append(summ_label)
				file_dist[veri_label] += 1
				summ_file_dist[summ_label] += 1
				label_dist[veri_label] += 1
				summ_label_dist[summ_label] += 1
			
	print(f'Test file: {test_file}')
	print('Statistics of training corpus:')
	print(f'Total non-rumors: {label_dist[0]}, Total rumors: {label_dist[1]}')
	print(f'Total non-summary-tweets: {summ_label_dist[0]}, Total summary-tweets: {summ_label_dist[1]}')
	weight_vec[test_file] = torch.tensor(compute_class_weight('balanced', numpy.unique(y), y)).to(device)
	summ_weight_vec[test_file] = torch.tensor(compute_class_weight('balanced', numpy.unique(summ_y), summ_y)).to(device)
	pos_weight = label_dist[0] / label_dist[1]
	pos_weight_vec[test_file] = torch.tensor([pos_weight], dtype=torch.float32).to(device)
	summ_pos_weight = summ_label_dist[0] / summ_label_dist[1]
	summ_pos_weight_vec[test_file] = torch.tensor([summ_pos_weight], dtype=torch.float32).to(device)
	print(f'Verification Class Weight Vector: {weight_vec[test_file]}')
	print(f'Verification Positive Class Weight Vector: {pos_weight_vec[test_file]}')
	print(f'Summary Class Weight Vector: {summ_weight_vec[test_file]}')
	print(f'Summary Positive Class Weight Vector: {summ_pos_weight_vec[test_file]}')



dfc = pd.read_pickle("MTLVS/data/features/dfc_0.57.pkl")
dfg = pd.read_pickle("MTLVS/data/features/dfg_0.72.pkl")
dfo = pd.read_pickle("MTLVS/data/features/dfo_0.6.pkl")
dfs = pd.read_pickle("MTLVS/data/features/dfs_0.6.pkl")


def get_numwords(x):
	tokens = [word for word in x['Orig_Tweet'].strip().split() if word not in string.punctuation]
	val1 = len(tokens)
	tokens = [word for word in x['Clean_Tweet'].strip().split() if word not in string.punctuation]
	val2 = len(tokens)
	tokens = [word for word in x['Norm_Tweet'].strip().split() if word not in string.punctuation]
	val3 = len(tokens)	
	val1 = 1 if math.isnan(val1) else val1
	val2 = 1 if math.isnan(val2) else val2
	val3 = 1 if math.isnan(val3) else val3
	return pd.Series([val1, val2, val3])


def generate_summary(numwords, alpha, dfsum):
	predicted_summary_orig = []
	predicted_summary_clean = []
	total_verified = 0
	later_verified = 0
	summary_length = 0
	summary_length_orig = 0
	summary_length_norm = 0
	count = 0
	
	dfpred1 = dfsum[dfsum['summ_pred']==1].reset_index(drop=True)
	dfpred1 = dfpred1[dfpred1['veri_pred']==0].reset_index(drop=True)
	dfpred1['Sorting_Criteria'] = (alpha * dfpred1['summ_pred_prob']) + ((1 - alpha) * dfpred1['veri_pred_prob'])
	if len(dfpred1) != 0:
		# dfpred1.sort_values(by=['summ_pred_prob'], ascending=False, inplace=True)
		dfpred1.sort_values(by=['Sorting_Criteria'], ascending=False, inplace=True)
		dfpred1.drop_duplicates(subset=['Clean_Tweet'], keep='first', inplace=True)
		dfpred1[["Num_words_orig", "Num_words_clean", "Num_words_norm"]] = dfpred1.apply(lambda x : get_numwords(x), axis= 1)
		l1 = len(dfpred1)
		for i, row in dfpred1.iterrows():
			if summary_length < numwords:
				count += 1
				predicted_summary_orig.append(row['Orig_Tweet'])
				predicted_summary_clean.append(row['Clean_Tweet'])
				if int(row['R1NR0']) == 0:
					total_verified += 1
				elif int(row['False0_True1_Unveri2_NR3_Rep4']) == 0:
					later_verified += 1
				summary_length += int(row['Num_words_clean'])
				summary_length_orig += int(row['Num_words_orig'])
				summary_length_norm += int(row['Num_words_norm'])
			else:
				break

	if summary_length < numwords:
		dfpred0 = dfsum[dfsum['summ_pred']==0].reset_index(drop=True)
		dfpred0 = dfpred0[dfpred0['veri_pred']==0].reset_index(drop=True)
		dfpred0['Sorting_Criteria'] = (alpha * (1 - dfpred0['summ_pred_prob'])) + ((1 - alpha) * dfpred0['veri_pred_prob'])
		if len(dfpred0) != 0:
			# dfpred0.sort_values(by=['summ_pred_prob'], ascending=True, inplace=True)
			dfpred0.sort_values(by=['Sorting_Criteria'], ascending=False, inplace=True)
			dfpred0.drop_duplicates(subset=['Clean_Tweet'], keep='first', inplace=True)
			dfpred0[["Num_words_orig", "Num_words_clean", "Num_words_norm"]] = dfpred0.apply(lambda x : get_numwords(x), axis= 1)
			l0 = len(dfpred0)
			for i, row in dfpred0.iterrows():
				if summary_length < numwords:
					count += 1
					predicted_summary_orig.append(row['Orig_Tweet'])
					predicted_summary_clean.append(row['Clean_Tweet'])
					if int(row['R1NR0']) == 0:
						total_verified += 1
					elif int(row['False0_True1_Unveri2_NR3_Rep4']) == 0:
						later_verified += 1
					summary_length += int(row['Num_words_clean'])
					summary_length_orig += int(row['Num_words_orig'])
					summary_length_norm += int(row['Num_words_norm'])
				else:
					break

	summ_orig = '.\n'.join(predicted_summary_orig).strip()
	summ_clean = '.\n'.join(predicted_summary_clean).strip()
	
	if count == 0:
		veri_prop = 0
		modified_veri_prop = 0
	else:
		veri_prop = float(total_verified / count)
		modified_veri_prop = float((total_verified + later_verified) / count)
	
	return summ_orig, summ_clean, veri_prop, modified_veri_prop, summary_length, summary_length_orig, summary_length_norm, count, total_verified, later_verified


def prepare_results(p, r, f):
	return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)



def train(tree_batch, test_file, mode="train"):
	loss = 0

	h, h_root, c, summ_out, att = model(
		tree_batch['f'].to(device),
		tree_batch['a'].to(device),
		tree_batch['k'].to(device),
		tree_batch['node_order'].to(device),
		tree_batch['adjacency_list'].to(device),
		tree_batch['edge_order'].to(device),
		tree_batch['root_node'].to(device)
	)
	# lastlayer_attention = att[-1]
	# lastlayer_attention = lastlayer_attention.to("cpu")
	# a = torch.mean(lastlayer_attention, dim=1)
	# cls_attentions = a[:, 0, :]
	# # token_attentions.append(cls_attentions)

	weights = weight_vec[test_file]
	pos_weights = pos_weight_vec[test_file]
	summ_weights = summ_weight_vec[test_file]
	summ_pos_weights = summ_pos_weight_vec[test_file]

	################################# Summary Loss ######################################
	
	# summ_gt_labels = tree_batch['summ_gt'].to('cpu')	
	# summ_gt_labels_tensor = torch.tensor(summ_gt_labels).type_as(summ_out).to(device)
	# summ_gt_labels_tensor = summ_gt_labels.clone().detach().type_as(summ_out).to(device)
	
	summ_logits = summ_out.detach().cpu()
	pred_v_summ, pred_labels_summ = torch.max(F.softmax(summ_logits, dim=1), 1)
	if SUMM_LOSS_FN == 'nw':
		loss_function_summ = CrossEntropyLoss()
	else:
		class_weights = summ_weights.float()
		loss_function_summ = CrossEntropyLoss(weight=class_weights)
	
	# loss_summ = loss_function_summ(summ_out, summ_gt_labels_tensor)
	loss_summ = loss_function_summ(summ_out, tree_batch['summ_gt'].long().to(device))

	#####################################################################################

	################################# Verification Loss #################################

	# g_labels = tree_batch['root_label'].to('cpu')
	# g_labels_tensor = torch.tensor(g_labels).type_as(h_root).to(device)	
	# g_labels_tensor = g_labels.clone().detach().type_as(h_root).to(device)
	
	pred_logits = h_root.detach().cpu()
	pred_v, pred_labels = torch.max(F.softmax(pred_logits, dim=1), 1)
	if VERI_LOSS_FN == 'nw':
		loss_function_veri = CrossEntropyLoss()
	else:
		class_weights = weights.float()
		loss_function_veri = CrossEntropyLoss(weight=class_weights)
	
	# loss_veri = loss_function_veri(h_root, g_labels_tensor.long())
	loss_veri = loss_function_veri(h_root, tree_batch['root_label'].long().to(device))

	#####################################################################################	

	#################################### VeriSumm Loss ##################################
	
	summ_gt_labels = tree_batch['summ_gt'].to('cpu')
	g_labels = tree_batch['root_label'].to('cpu')
	# batch_size = len(summ_gt_labels.tolist())
	# indices = [i for i in range(batch_size) if summ_gt_labels[i] == 1 and pred_labels_summ[i] == 1 and g_labels[i] == 0 and pred_labels[i] == 1]
	# print(indices)
	# veri_pred = torch.tensor([h_root[i].tolist() for i in indices]).to(device)
	# veri_gt = torch.tensor([g_labels[i].tolist() for i in indices]).to(device)
	# loss_verisumm = torch.tensor(0).to(device)
	# if len(indices):
	# 	loss_verisumm = loss_function_veri(veri_pred, veri_gt.long())
		
	
	#################################### Combined Loss ##################################
	loss = LAMBDA1*loss_veri + LAMBDA2*loss_summ
	# loss = LAMBDA1*loss_veri + LAMBDA2*loss_summ + LAMBDA3*loss_verisumm
	
	if FLOOD == 'y':
		loss = (loss-0.25).abs() + 0.25	

	optimizer.zero_grad()
	
	if mode == "train":
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()
	
	return loss, pred_labels, g_labels, pred_labels_summ, summ_gt_labels, loss_veri, loss_summ


if TEST_FILE.startswith('charlie'):
	dftest = dfc.copy(deep=False)
elif TEST_FILE.startswith('german'):
	dftest = dfg.copy(deep=False)
elif TEST_FILE.startswith('ottawa'):
	dftest = dfo.copy(deep=False)
else:
	dftest = dfs.copy(deep=False)

dftest.Tweet_ID = dftest.Tweet_ID.astype(int)


def get_data(x):	
	orig_tweet_text = dftest.loc[dftest.Tweet_ID==x['Tweet_ID'], 'Orig_Tweet'].values[0]
	clean_tweet_text = dftest.loc[dftest.Tweet_ID==x['Tweet_ID'], 'Clean_Tweet'].values[0]
	norm_tweet_text = dftest.loc[dftest.Tweet_ID==x['Tweet_ID'], 'Norm_Tweet'].values[0]
	orig_summ_gt = dftest.loc[dftest.Tweet_ID==x['Tweet_ID'], 'Summary_gt'].values[0]
	new_summ_gt = dftest.loc[dftest.Tweet_ID==x['Tweet_ID'], 'New_Summ_gt_Clean'].values[0]
	r1nr0 = dftest.loc[dftest.Tweet_ID==x['Tweet_ID'], 'R1NR0'].values[0]
	veracity = dftest.loc[dftest.Tweet_ID==x['Tweet_ID'], 'False0_True1_Unveri2_NR3_Rep4'].values[0]

	return pd.Series([orig_tweet_text, clean_tweet_text, norm_tweet_text, orig_summ_gt, new_summ_gt, r1nr0, veracity])


for lr in lr_list:
	print("\n\n\nTraining with LR: ", lr)
	
	for test in files:
		if not test.startswith(TEST_FILE):
			continue
		
		# path = "./Models/"
				
		# name = path + "mtl_verification_" + TEST_FILE + "_feat" + MODEL_NAME + ".pt"
		# name = path + "mtl_" + TEST_FILE + ".pt"
		
		tweetconfig.output_attentions = True
		tweetconfig.output_hidden_states = True				
		model = TreeLSTM(MODEL_NAME, TRAINABLE_LAYERS, IN_FEATURES, OUT_FEATURES, CLASSIFIER_DROPOUT, mode="cls", tweetconfig=tweetconfig)
		model.cuda(gpu_id)
		
		# test_model = TreeLSTM(MODEL_NAME, TRAINABLE_LAYERS, IN_FEATURES, OUT_FEATURES, CLASSIFIER_DROPOUT, mode="cls", tweetconfig=tweetconfig)
		# test_model.cuda(gpu_id)
				
		if OPTIM == 'adam':
			if L2_REGULARIZER == 'n':
				optimizer = torch.optim.Adam(model.parameters(), lr=lr)
			else:
				print(f"L2_REGULARIZER = y and WEIGHT_DECAY = {WEIGHT_DECAY}")
				optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
		else:
			optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)

		# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, verbose=True)		

		test_file = test
		print('Training Set:', set(files) - {test_file})
		test_trees = []
		train_trees = []
		val_data = []
		for filename in files:
			if filename == test:
				test_trees.extend(tree_li[filename])
				test_trees.extend(val_li[filename])
			else:
				curr_tree_dataset = TreeDataset(tree_li[filename])
				train_trees.extend(curr_tree_dataset)
				val_data.extend(TreeDataset(val_li[filename]))
		
		print("size of training data", len(train_trees))
		print("Size of test data", len(test_trees))		
		print("\ntraining started....")
		
		prev_loss = math.inf
		prev_acc = 0
		best_epoch = 0

		for i in range(NUM_ITERATIONS):

			data_gen = DataLoader(
				train_trees,
				collate_fn=batch_tree_input,
				batch_size=BATCH_SIZE,
				shuffle = True)

			val_gen = DataLoader(
				val_data,
				collate_fn=batch_tree_input,
				batch_size=BATCH_SIZE,
				shuffle = False)
			
			model.train()
			predicted_labels = []
			ground_labels = []			
			summ_predicted_labels = []
			summ_ground_labels = []			
			j = 0
			train_avg_loss=0
			train_avg_veri_loss = 0
			train_avg_summ_loss = 0					
			for tree_batch in data_gen:
				loss, p_labels, g_labels, p_summ_labels, summ_gt_labels, v_loss, s_loss = train(tree_batch, test_file, "train")
				predicted_labels.extend(p_labels)
				ground_labels.extend(g_labels)
				summ_predicted_labels.extend(p_summ_labels)
				summ_ground_labels.extend(summ_gt_labels)				
				j = j+1
				train_avg_loss += loss
				train_avg_veri_loss += v_loss
				train_avg_summ_loss += s_loss
			veri_train_acc = accuracy_score(ground_labels, predicted_labels)
			summ_train_acc = accuracy_score(summ_ground_labels, summ_predicted_labels)
			train_avg_loss /= j
			train_avg_veri_loss /= j
			train_avg_summ_loss /= j
			
			print("validation started..", len(val_data))
			model.eval()
			val_predicted_labels= []
			val_ground_labels = []
			val_summ_predicted_labels = []			
			val_summ_ground_labels = []			
			val_j = 0
			val_avg_loss = 0
			val_avg_veri_loss = 0
			val_avg_summ_loss = 0
			with torch.no_grad():
				for batch in val_gen:
					loss, p_labels, g_labels, p_summ_labels, summ_gt_labels, v_loss, s_loss = train(batch, test_file, "eval")
					val_predicted_labels.extend(p_labels)
					val_ground_labels.extend(g_labels)
					val_summ_predicted_labels.extend(p_summ_labels)
					val_summ_ground_labels.extend(summ_gt_labels)					
					val_j += 1
					val_avg_loss += loss
					val_avg_veri_loss += v_loss
					val_avg_summ_loss += s_loss
			veri_val_acc = accuracy_score(val_ground_labels, val_predicted_labels)
			veri_val_f1 = f1_score(val_ground_labels, val_predicted_labels, average='macro')
			summ_val_acc = accuracy_score(val_summ_ground_labels, val_summ_predicted_labels)
			summ_val_f1 = f1_score(val_summ_ground_labels, val_summ_predicted_labels, average='macro')
			val_avg_loss /= val_j
			val_avg_veri_loss /= val_j
			val_avg_summ_loss /= val_j
			
			if MODEL_SAVING_POLICY == "acc":
				if(prev_acc <= veri_val_acc):
					# save_model(model, name, veri_val_acc, val_avg_loss)
					prev_acc = veri_val_acc
					best_epoch = i
			else:			
				# if(prev_loss >= val_avg_loss):
				# 	save_model(model, name, veri_val_acc, val_avg_loss)
				# 	prev_loss = val_avg_loss
				if(prev_loss >= val_avg_veri_loss):
					prev_loss = val_avg_veri_loss					
					best_epoch = i
			
			print('\nIteration ', i)
			print('Training Joint Loss: ', train_avg_loss)
			print('Training Verification Loss: ', train_avg_veri_loss)
			print('Training Summarization Loss: ', train_avg_summ_loss)
			print('Training Verification accuracy: ', veri_train_acc)
			print('Training Summarization accuracy: ', summ_train_acc)
			print('\nValidation loss: ', val_avg_loss)
			print('Validation Verification Loss: ', val_avg_veri_loss)
			print('Validation Summarization Loss: ', val_avg_summ_loss)
			print('Validation Verification accuracy: ', veri_val_acc)
			print('Validation Summarization accuracy: ', summ_val_acc)
			print('Validation Verification macro f1 score: ', veri_val_f1)
			print('Validation Summarization macro f1 score: ', summ_val_f1)
			
			# scheduler.step(veri_val_acc)

		print('Training Complete')		
		print(f'\n\nNow training the model with (train + validation) data for {best_epoch} no. of epochs.')
		
		model = TreeLSTM(MODEL_NAME, TRAINABLE_LAYERS, IN_FEATURES, OUT_FEATURES, CLASSIFIER_DROPOUT, mode="cls", tweetconfig=tweetconfig)
		model.cuda(gpu_id)
				
		if OPTIM == 'adam':
			if L2_REGULARIZER == 'n':
				optimizer = torch.optim.Adam(model.parameters(), lr=lr)
			else:
				print(f"L2_REGULARIZER = y and WEIGHT_DECAY = {WEIGHT_DECAY}")
				optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
		else:
			optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)

		all_train_trees = []		
		for filename in files:
			if filename != test:				
				all_train_trees.extend(TreeDataset(tree_li[filename]))
				all_train_trees.extend(TreeDataset(val_li[filename]))		
		print("size of (train + validation) data: ", len(all_train_trees))
		
		for i in range(best_epoch+1):
			data_gen = DataLoader(
				all_train_trees,
				collate_fn=batch_tree_input,
				batch_size=BATCH_SIZE,
				shuffle = True)

			model.train()
			predicted_labels = []
			ground_labels = []			
			summ_predicted_labels = []
			summ_ground_labels = []			
			j = 0
			train_avg_loss=0
			train_avg_veri_loss = 0
			train_avg_summ_loss = 0					
			for tree_batch in data_gen:
				loss, p_labels, g_labels, p_summ_labels, summ_gt_labels, v_loss, s_loss = train(tree_batch, test_file, "train")
				predicted_labels.extend(p_labels)
				ground_labels.extend(g_labels)
				summ_predicted_labels.extend(p_summ_labels)
				summ_ground_labels.extend(summ_gt_labels)				
				j = j+1
				train_avg_loss += loss
				train_avg_veri_loss += v_loss
				train_avg_summ_loss += s_loss
			veri_train_acc = accuracy_score(ground_labels, predicted_labels)
			summ_train_acc = accuracy_score(summ_ground_labels, summ_predicted_labels)
			train_avg_loss /= j
			train_avg_veri_loss /= j
			train_avg_summ_loss /= j

		print('Training Joint Loss: ', train_avg_loss)
		print('Training Verification Loss: ', train_avg_veri_loss)
		print('Training Summarization Loss: ', train_avg_summ_loss)
		print('Training Verification accuracy: ', veri_train_acc)
		print('Training Summarization accuracy: ', summ_train_acc)

		# if ((i+1) % 5 == 0 and i > 0):		
		# load_model(test_model, name)
		print('Now Testing:', test_file)
		total = 0
		tweet_ids = []
		veri_predicted = []
		veri_ground = []
		veri_prob = []
		summ_predicted = []
		summ_ground = []
		summ_prob = []
		token_attentions = []
		num_tokens = []
		tokenslist = []

		model.eval()
		with torch.no_grad():
			for test in test_trees:						
				try:
					h_test, h_test_root, c, summ_out, att = model(
							test['f'].to(device),
							test['a'].to(device),
							test['k'].to(device),
							test['node_order'].to(device),
							test['adjacency_list'].to(device),
							test['edge_order'].to(device),
							test['root_n'].to(device)
					)
				except:
					continue
				# lastlayer_attention = att[-1][0]
				# lastlayer_attention = lastlayer_attention.to("cpu")
				# # print("shape of cls:",lastlayer_attention.shape)
				# a = torch.mean(lastlayer_attention, dim=0).squeeze(0)
				# # print("after mean:",a.shape)
				# cls_attentions = a[0]
				# token_attentions.append(cls_attentions)
				# tokens = tokenizer.convert_ids_to_tokens(test['f'][0])
				# tokenslist.append(tokens)
				# num_tokens.append(int(torch.sum(test['a'][0]).item()))

				tweet_ids.append(test['tweet_id'].item())

				g_label = test['root_l'].to('cpu')
				pred_logits = h_test_root.detach().cpu()
				pred_v, pred_label = torch.max(F.softmax(pred_logits, dim=1), 1)
				veri_predicted.append(pred_label.item())
				veri_prob.append(pred_v.item())
				veri_ground.append(g_label.item())		

				summ_gt_label = test['summ_gt'].to('cpu')	
				summ_logits = summ_out.detach().cpu()
				pred_v_summ, pred_label_summ = torch.max(F.softmax(summ_logits, dim=1), 1)
				summ_predicted.append(pred_label_summ.item())
				summ_prob.append(pred_v_summ.item())						
				summ_ground.append(summ_gt_label.item())					

				total += 1
		
		print(f'\nTotal Test trees evaluated: {total}')
		accuracy = accuracy_score(veri_ground, veri_predicted)
		print('Accuracy: %f' % accuracy)
		precision = precision_score(veri_ground, veri_predicted)
		print('Precision: %f' % precision)
		precision = precision_score(veri_ground, veri_predicted, average='macro')
		print('Macro Precision: %f' % precision)
		recall = recall_score(veri_ground, veri_predicted)
		print('Recall: %f' % recall)
		recall = recall_score(veri_ground, veri_predicted, average='macro')
		print('Macro Recall: %f' % recall)
		f1 = f1_score(veri_ground, veri_predicted)
		print('Micro F1 score: %f' % f1)
		f1 = f1_score(veri_ground, veri_predicted, average='macro')
		print('Macro F1 score: %f' % f1)
		print("\n\n")
		print(classification_report(veri_ground, veri_predicted, digits=5))
		print("\n\n")
		print('confusion matrix ', confusion_matrix(veri_ground, veri_predicted))

		dfsum = pd.DataFrame({
						"Tweet_ID": tweet_ids, 
						"veri_pred": veri_predicted, 
						"veri_pred_prob": veri_prob, 
						"veri_gt": veri_ground, 
						"summ_pred": summ_predicted, 
						"summ_pred_prob": summ_prob, 
						"summ_gt": summ_ground}
						# "Tokens": tokenslist, 
						# "Attentions": token_attentions, 
						# "Numtokens": num_tokens}
					)
		
		dfsum[["Orig_Tweet", "Clean_Tweet", "Norm_Tweet", "Summary_gt", "New_Summary_gt", "R1NR0", "False0_True1_Unveri2_NR3_Rep4"]] = dfsum.apply(lambda x : get_data(x), axis= 1)

		# assert dfsum['Situational'].values.tolist() == [int(tag) for tag in dfsum['sit_tag'].values.tolist()]
		assert dfsum['New_Summary_gt'].values.tolist() == dfsum['summ_gt'].values.tolist()
		assert dfsum['R1NR0'].values.tolist() == dfsum['veri_gt'].values.tolist()
						
		dfsum.sort_values(by=['summ_pred', 'summ_pred_prob'], inplace=True, ascending=False)
		dfsum = dfsum.reset_index(drop=True)

		groundtruth_summary_original = ""
		glist_original = dfsum[dfsum['Summary_gt']==1]['Orig_Tweet'].values
		groundtruth_summary_original = ".\n".join(glist_original)

		groundtruth_summary_cleaned  = ""		
		glist_cleaned = dfsum[dfsum['Summary_gt']==1]['Clean_Tweet'].values
		groundtruth_summary_cleaned = ".\n".join(glist_cleaned)
		
		# if L2_REGULARIZER == 'n':
		# 	# dfsum.to_pickle("MTL_" + TEST_FILE + "_"  + MODEL_NAME + "_" + str(IN_FEATURES) + "_SIT_" + str(USE_SITU) + "_L2_n_LR_" + str(lr) + "_LAM1_" + str(LAMBDA1) + "_LAM2_" + str(LAMBDA2) + ".pkl")
		# 	dfsum.to_pickle("MTL_" + TEST_FILE + "_"  + MODEL_NAME + "_" + str(IN_FEATURES) + "_L2_n_LR_" + str(lr) + "_LAM1_" + str(LAMBDA1) + "_LAM2_" + str(LAMBDA2) + ".pkl")
		# else:
		# 	# dfsum.to_pickle("MTL_" + TEST_FILE + "_"  + MODEL_NAME + "_" + str(IN_FEATURES) + "_SIT_" + str(USE_SITU) + "_L2_y_" + str(WEIGHT_DECAY) + "_LR_" + str(lr) + "_LAM1_" + str(LAMBDA1) + "_LAM2_" + str(LAMBDA2) + ".pkl")
		# 	dfsum.to_pickle("MTL_" + TEST_FILE + "_"  + MODEL_NAME + "_" + str(IN_FEATURES) + "_L2_y_" + str(WEIGHT_DECAY) + "_LR_" + str(lr) + "_LAM1_" + str(LAMBDA1) + "_LAM2_" + str(LAMBDA2) + ".pkl")

		for alpha in [0, 0.5, 1]:
			print(f'\n\nFor alpha={alpha}')
			summ_orig, summ_clean, veri_prop, modified_veri_prop, summary_length, summary_length_orig, summary_length_norm, count, total_verified, later_verified = generate_summary(250, alpha, dfsum)
			print(f'\nSummary generated for: {TEST_FILE}')
			print(f'Total tweets: {count}')
			print(f'Total verified: {total_verified}')
			print(f'Total later verified: {later_verified}')
			print(f'Summary length with clean tweets: {summary_length}')
			print(f'Summary length with original tweets: {summary_length_orig}')
			print(f'Summary length with normalized tweets: {summary_length_norm}')
			print(f'Verified_Ratio of tweets: {veri_prop}')
			print(f'Modified verified_Ratio of tweets: {modified_veri_prop}\n\n')

			with open(TEST_FILE + "_test_orig_250.txt",'w') as f:
				f.write(summ_orig)

			with open(TEST_FILE + "_test_clean_250.txt",'w') as f:
				f.write(summ_clean)

			# content = ""
			# content = content + "groundtruth_summary_original : \n\n" + groundtruth_summary_original + ".\n\n"
			# content = content + "\ngroundtruth_summary_cleaned : \n\n" + groundtruth_summary_cleaned + ".\n\n"
			# content = content + f'\nGenerated Summary Length = {summary_length}\n'
			# content = content + "\nsummary_original_250 : \n\n" + summ_orig + ".\n\n"
			# content = content + "\nsummary_cleaned_250 : \n\n" + summ_clean + ".\n\n"
							
			# if L2_REGULARIZER == 'y':
			# 	summary_filepath = "MTL_" + TEST_FILE + "_"  + MODEL_NAME + "_" + str(IN_FEATURES) + "_L2_y_" + str(WEIGHT_DECAY) + "_LR_" + str(lr) + "_LAM1_" + str(LAMBDA1) + "_LAM2_" + str(LAMBDA2) + "_summary.txt"
			# else:
			# 	summary_filepath = "MTL_" + TEST_FILE + "_"  + MODEL_NAME + "_" + str(IN_FEATURES) + "_L2_n_LR_" + str(lr) + "_LAM1_" + str(LAMBDA1) + "_LAM2_" + str(LAMBDA2) + "_summary.txt"
			
			# with open(summary_filepath, 'w') as f:
			# 	f.write(content)

			sumdict = {"Original_summary": [groundtruth_summary_original, summ_orig],
						"Cleaned_summary": [groundtruth_summary_cleaned, summ_clean]}

			for summ, hyp in sumdict.items():
				print("\n\nMetric for: ", summ)
				for aggregator in ['Avg', 'Best', 'Individual']:
					print('Evaluation with {}'.format(aggregator))
					apply_avg = aggregator == 'Avg'
					apply_best = aggregator == 'Best'

					evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
											max_n=2,
											# limit_length=True,
											# length_limit=100,
											length_limit_type='words',
											apply_avg=apply_avg,
											apply_best=apply_best,
											alpha=0.5,  # Default F1_score
											weight_factor=1.2,
											stemming=True)

					all_hypothesis = [hyp[1]]
					all_references = [hyp[0]]

					scores = evaluator.get_scores(all_hypothesis, all_references)

					for metric, results in sorted(scores.items(), key=lambda x: x[0]):
						if not apply_avg and not apply_best:  # value is a type of list as we evaluate each summary vs each reference
							for hypothesis_id, results_per_ref in enumerate(results):
								nb_references = len(results_per_ref['p'])
								for reference_id in range(nb_references):
									print('\tHypothesis #{} & Reference #{}: '.format(
										hypothesis_id, reference_id))
									print('\t' + prepare_results(results_per_ref['p'][reference_id],
																results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
							print()
						else:
							print(prepare_results(results['p'], results['r'], results['f']))
					print()

			

			print("\n\nFor Original Summary:")
			if summ_orig.strip() != "":
				os.chdir("twitie-tagger")
				return_code = subprocess.call("/usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java -jar twitie_tag.jar models/gate-EN-twitter.model '../" + TEST_FILE + "_test_orig_250.txt' > '../" + TEST_FILE + "_output_orig_250.txt'", shell=True)
				# return_code = subprocess.call("java -jar twitie_tag.jar models/gate-EN-twitter.model '../" + TEST_FILE + "_test_orig_250.txt' > '../" + TEST_FILE + "_output_orig_250.txt'", shell=True)
				os.chdir("..")

				num_words = 0
				content_words =0
				with open(TEST_FILE + "_output_orig_250.txt", 'r') as f:
					for line in f:
						words = line.split()
						for word in words:
							l = word.split("_")
							if l[-1].startswith("CD") or l[-1].startswith("NN") or l[-1].startswith("VB"):
								content_words+=1
						num_words += len(words)
				print("Number of words: " , num_words)
				print("Number of content words: " ,	content_words)	
				print("Ratio :" , content_words/num_words)
			else:
				print("Number of words: " , 0)
				print("Number of content words: " ,	0)	
				print("Ratio :" , 0)

			print("\n\nFor Cleaned Summary:")
			if summ_clean.strip() != "":
				os.chdir("twitie-tagger")
				return_code = subprocess.call("/usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java -jar twitie_tag.jar models/gate-EN-twitter.model '../" + TEST_FILE + "_test_clean_250.txt' > '../" + TEST_FILE + "_output_clean_250.txt'", shell=True)  
				# return_code = subprocess.call("java -jar twitie_tag.jar models/gate-EN-twitter.model '../" + TEST_FILE + "_test_clean_250.txt' > '../" + TEST_FILE + "_output_clean_250.txt'", shell=True)  
				os.chdir("..")

				num_words = 0
				content_words =0
				with open(TEST_FILE + "_output_clean_250.txt", 'r') as f:
					for line in f:
						words = line.split()
						for word in words:
							l = word.split("_")
							if l[-1].startswith("CD") or l[-1].startswith("NN") or l[-1].startswith("VB"):
								content_words += 1
						num_words += len(words)
				print("Number of words: " , num_words)
				print("Number of content words: " , content_words)
				print("Ratio :" , content_words/num_words)
			else:
				print("Number of words: " , 0)
				print("Number of content words: " ,	0)	
				print("Ratio :" , 0)

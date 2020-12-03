import os
import sys
import math
import copy
import argparse
import codecs
import random
import numpy
import itertools
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import *
# import matplotlib.pyplot as plt
# from tqdm import tqdm

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu_id', type=int, default=0)
	parser.add_argument('--model_name', type=str, default="BERT") 
	parser.add_argument('--in_features', type=int, default=768)
	parser.add_argument('--save_policy', type=str, default="loss")
	parser.add_argument('--loss_fn', type=str, default="nw")
	parser.add_argument('--optim', type=str, default="adam")
	parser.add_argument('--l2', type=str, default="y")
	parser.add_argument('--wd', type=float, default=0.01)
	parser.add_argument('--use_dropout', type=str, default="n")
	parser.add_argument('--classifier_dropout', type=float, default=0.4)
	# parser.add_argument('--tree', type=str, default="new")
	parser.add_argument('--events', type=int, default=4)
	parser.add_argument('--iters', type=int, default=5)
	parser.add_argument('--bs', type=int, default=16)
	parser.add_argument('--seed', type=int, default=1955)
	parser.add_argument('--test_file', type=str, default="german")

	args = parser.parse_args()

	gpu_id = args.gpu_id
	print(f'GPU_ID = {gpu_id}\n')	
	MODEL_NAME = args.model_name
	print(f'MODEL_NAME = {MODEL_NAME}')
	IN_FEATURES = args.in_features
	print(f'IN_FEATURES = {IN_FEATURES}')
	OUT_FEATURES = 128
	print(f'OUT_FEATURES = {OUT_FEATURES}')
	MODEL_SAVING_POLICY = args.save_policy
	print(f'MODEL_SAVING_POLICY = {MODEL_SAVING_POLICY}')
	LOSS_FN = args.loss_fn
	print(f'LOSS_FN = {LOSS_FN}')
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
	# TREE_VERSION = args.tree
	# print(f'TREE_VERSION = {TREE_VERSION}')
	NO_OF_EVENTS = args.events
	print(f'NO_OF_EVENTS = {NO_OF_EVENTS}')
	NUM_ITERATIONS = args.iters
	print(f'NUM_ITERATIONS = {NUM_ITERATIONS}')
	BATCH_SIZE = args.bs
	print(f'BATCH_SIZE = {BATCH_SIZE}')
	lr_list = [5e-6, 1e-5, 2e-5, 5e-5, 1e-4]
	print(f'LEARNING_RATES = {str(lr_list)}')
	TRAINABLE_LAYERS = [0,1,2,3,4,5,6,7,8,9,10,11]
	print(f'TRAINABLE_LAYERS = {str(TRAINABLE_LAYERS)}')
	TEST_FILE = args.test_file
	print(f'\nTEST_FILE = {TEST_FILE}')

	# g_cpu = torch.Generator()
	# seed_val = g_cpu.seed()
	seed_val = args.seed
	print(f'\nSEED = {str(seed_val)}\n\n')



# # If there's a GPU available...
if torch.cuda.is_available():
	# Tell PyTorch to use the GPU.
	if gpu_id == 0:
		device = torch.device("cuda:0")
	else:
		device = torch.device("cuda:1")
	print('There are %d GPU(s) available.' % torch.cuda.device_count())
	print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
	print('No GPU available, using the CPU instead.')
	device = torch.device("cpu")



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)
# To make results reproducable
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
	root_label = [labels[0]]
	adjacency_list = _gather_adjacency_list(tree)

	node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(features))
	root_node = [0]

	return {
		'f': torch.tensor(features, dtype=torch.long),
		'a':torch.tensor(attention,  dtype=torch.float32),
		'k':torch.tensor(old_features, dtype=torch.float32),
		'l': torch.tensor(labels,  dtype=torch.float32),
		'root_l': torch.tensor(root_label, dtype=torch.long),
		'root_n': torch.tensor(root_node,  dtype=torch.int64),
		'node_order': torch.tensor(node_order,  dtype=torch.int64),
		'adjacency_list': torch.tensor(adjacency_list,  dtype=torch.int64),
		'edge_order': torch.tensor(edge_order,  dtype=torch.int64),
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
	
	# print(adjacency_list)
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

	batched_root_labels = torch.cat([b['root_l'] for b in batch])
	
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
		'l': batched_labels
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
		self.model_name = model_name
		
		if model_name == 'BERT':
			self.BERT_model = BertModel.from_pretrained("bert-base-cased", output_attentions=True)
		elif model_name == 'ROBERTA':
			self.BERT_model = RobertaModel.from_pretrained("roberta-base", output_attentions=True)
		elif model_name == 'BERTWEET':
			self.BERT_model = RobertaModel.from_pretrained("MTLVS/BERTweet_base_transformers/model.bin", config=tweetconfig)
		
		# elif model_name == 'XLNET':
		# 	self.BERT_model = XLNetModel.from_pretrained("xlnet-base-cased")
		# elif model_name == 'T5':
		# 	self.BERT_model = T5Model.from_pretrained("t5-base")
				
		
		# for name, param in self.BERT_model.named_parameters():
		# 	flag = False
		# 	for num in trainable_layers:
		# 		if 'layer.'+ str(num) + '.' in name:
		# 			param.requires_grad = True
		# 			flag = True
		# 			break
		# 	if not flag:
		# 		if 'pooler' in name or 'embedding' in name:
		# 			param.requires_grad = True
		# 		else:
		# 			param.requires_grad = False

		# self.W_iou = torch.nn.Linear(self.in_features, 3 * self.out_features)
		# self.U_iou = torch.nn.Linear(self.out_features, 3 * self.out_features, bias=False)

		# f terms are maintained seperate from the iou terms because they involve sums over child nodes
		# while the iou terms do not
		# self.W_f = torch.nn.Linear(self.in_features, self.out_features)
		# self.U_f = torch.nn.Linear(self.out_features, self.out_features, bias=False)
		
		self.fc1 = torch.nn.Linear(self.in_features, self.out_features)
		# self.fc = torch.nn.Linear(self.out_features, 2)		
		self.fc2 = torch.nn.Linear(self.out_features, 1)
		
		# self.bert_dropout = torch.nn.Dropout(bert_dropout)
		
		self.classifier_dropout = torch.nn.Dropout(classifier_dropout)
		
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
		
		# if self.model_name == 'XLNET':
		# 	hidden_states = self.BERT_model(input_ids=features, attention_mask=attentions)
		# 	hidden_states = hidden_states[0]
		# 	# print(len(hidden_states))
		# 	print(hidden_states[0])
		# 	# print(hidden_states[0].size)
		# else:
		
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
		
		# for n in range(node_order.max() + 1):
		# 	self._run_lstm(n, h, c, output_vectors, node_order, adjacency_list, edge_order)

		# h_root = h[root_node, :]
		out1 = self.fc1(output_vectors[root_node,:])
		
		if USE_DROPOUT == 'y':
			# h_root = self.classifier_dropout(h_root)
			out1 = self.classifier_dropout(out1)
		
		# logits_out = self.fc(h_root)
		logits_out = self.fc2(out1)
		
		# pred_out = F.log_softmax(logits_out, dim = 1)
		# pred_out = F.softmax(logits_out, dim = 1)
		
		return h, logits_out, c, att	


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


def split_data(trees, frac):
	pos_data = []
	neg_data = []
	for tree in trees:
		if tree['root_l'].tolist() == [[0, 1]]:
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


if MODEL_NAME == 'BERT':
	tree_path = 'MTLVS/data/features/PT_PHEME5_FeatBERT40_Depth5_maxR5_MTL_Final/'
elif MODEL_NAME == 'ROBERTA':
	tree_path = 'MTLVS/data/features/PT_PHEME5_FeatROBERTA40_Depth5_maxR5_MTL_Final/'
elif MODEL_NAME == 'BERTWEET':
	tree_path = 'MTLVS/data/features/PT_PHEME5_FeatBERTWEET40_Depth5_maxR5_MTL_Final/'

if NO_OF_EVENTS == 4:
	files = ['charliehebdo.txt', 'germanwings-crash.txt', 'ottawashooting.txt','sydneysiege.txt']
else:
	files = ['charliehebdo.txt', 'ferguson.txt', 'germanwings-crash.txt', 'ottawashooting.txt','sydneysiege.txt']


tree_li = {}
val_li = {}
for filename in files:
	input_file = codecs.open(tree_path + filename, 'r', 'utf-8')
	tree_li[filename]=[]
	for row in input_file:
		s = row.strip().split('\t')
		tweet_id = int(s[0])
		curr_tree = eval(s[1])
		# try:
		curr_tensor = convert_tree_to_tensors(curr_tree, tweet_id)
		# except Exception as e:
		#     # print(e)
		#     continue
		tree_li[filename].append(curr_tensor)
		# tree_li.append(curr_tree)

	random.shuffle(tree_li[filename])
	# val_len = int(0.1*len(tree_li[filename]))
	# val_li[filename] = (tree_li[filename][:val_len])
	# tree_li[filename] = tree_li[filename][val_len:] 
	tree_li[filename], val_li[filename] = split_data(tree_li[filename], 0.1)
	input_file.close()
	print(f'{filename} Training Set Size: {len(tree_li[filename])}, Validation Set Size: {len(val_li[filename])}, Total: {len(tree_li[filename]) + len(val_li[filename])}')


from sklearn.utils.class_weight import compute_class_weight
weight_vec = {}
pos_weight_vec = {}
for test_file in files:
	y = []
	label_dist = [0, 0]
	for filename in files:		
		if filename != test_file:			
			file_dist = [0, 0]
			for tree in tree_li[filename]:
				# print(int(tree['root_l'].tolist()[0][1]))
				y.append(int(tree['root_l'].tolist()[0][1]))
				file_dist[int(tree['root_l'].tolist()[0][1])] += 1
				label_dist[int(tree['root_l'].tolist()[0][1])] += 1
			
	print(f'Total non-rumors: {label_dist[0]}, Total rumors: {label_dist[1]}')
	weight_vec[test_file] = torch.tensor(compute_class_weight('balanced', numpy.unique(y), y)).to(device)
	pos_weight = label_dist[0] / label_dist[1]
	pos_weight_vec[test_file] = torch.tensor([pos_weight], dtype=torch.float32).to(device)
	print(f'Test File: {test_file}, Weight Vector: {weight_vec[test_file]}')
	print(f'Test File: {test_file}, Pos Weight Vector: {pos_weight_vec[test_file]}')



def train(tree_batch, test_file, mode="train"):
	err_count = 0
	loss = 0
	pred_labels = []
	g_labels = []
	
	# try:
	h, h_root, c, att = model(
		tree_batch['f'].to(device),
		tree_batch['a'].to(device),
		tree_batch['k'].to(device),
		tree_batch['node_order'].to(device),
		tree_batch['adjacency_list'].to(device),
		tree_batch['edge_order'].to(device),
		tree_batch['root_node'].to(device)
	)

	# print(h_root)
	# print(h_root.size())
	# print(f'Predicted Label Type: {h_root.dtype}')
		
	weights = weight_vec[test_file]
	pos_weights = pos_weight_vec[test_file]

	# CASE 1: 2 o/p neurons, BCEWithLogitsLoss/BCELoss
	# root_labels = tree_batch['root_label'].to(device)
	# print(root_labels)
	# print(f'Root Label Type: {root_labels.dtype}')
	# root = root_labels.to('cpu')
	# g_labels = [t[1] for t in root]
	# pred_logits = h_root.detach().cpu()
	# For getting the predicted label, it's immaterial whether we apply softmax or not
	# pred_v, pred_labels = torch.max(F.softmax(pred_logits, dim=1), 1)
	# pred_v, pred_labels = torch.max(pred_logits, 1)
	# loss_function = torch.nn.BCEWithLogitsLoss()
	# loss_function = torch.nn.BCEWithLogitsLoss(weight=weights)
	# loss = loss_function(h_root, root_labels)

	# CASE 2: 2 o/p neurons, CrossEntropyLoss
	# root = tree_batch['root_label'].to('cpu')
	# g_labels = [t[1] for t in root]
	# g_labels_tensor = torch.tensor(g_labels).to(device)
	# pred_logits = h_root.detach().cpu()
	# For getting the predicted label, it's immaterial whether we apply softmax or not
	# pred_v, pred_labels = torch.max(F.softmax(pred_logits, dim=1), 1)
	# pred_v, pred_labels = torch.max(pred_logits, 1)
	# loss_function = torch.nn.CrossEntropyLoss(weight=weights)
	# loss_function = torch.nn.CrossEntropyLoss()
	# loss = loss_function(h_root, g_labels_tensor)

	# CASE 3: 1 o/p neuron, BCEWithLogitsLoss
	root = tree_batch['root_label'].to('cpu')
	# print(root)
	g_labels = [[t[1]] for t in root]
	# print(g_labels)
	g_labels_tensor = torch.tensor(g_labels).type_as(h_root).to(device)
	# print(g_labels_tensor)
	# print(g_labels_tensor.size())
	pred_logits = h_root.detach().cpu()
	# print(pred_logits)
	logits_after_sigmoid = sigmoid_fn(pred_logits)
	# print(logits_after_sigmoid)
	batch_size = logits_after_sigmoid.size()[0]		
	pred_labels = [1 if logits_after_sigmoid[i].item() >= 0.5 else 0 for i in range(batch_size)]
	# print(pred_labels)
	pred_labels = torch.tensor(pred_labels)
	# print(pred_labels)
	
	if LOSS_FN == 'nw':
		loss_function = torch.nn.BCEWithLogitsLoss()
	else:
		loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
	# loss_function = torch.nn.BCEWithLogitsLoss(weight=weights)
	loss = loss_function(h_root, g_labels_tensor)
	
	g_labels = [t[1] for t in root]
	
	optimizer.zero_grad()
	
	if mode == "train":
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()
	
	# except Exception as e:
	#     print("error with tree: ",e)
	#     err_count = 1
	
	return loss, pred_labels, g_labels, err_count


for lr in lr_list:
	print("\n\n\nTraining with LR: ", lr)
	train_accuracy = []
	val_accuracy = []
	for test in files:
		if not test.startswith(TEST_FILE):
			continue
		
		# random.seed(seed_val)
		# numpy.random.seed(seed_val)
		# torch.manual_seed(seed_val)
		# torch.cuda.manual_seed(seed_val)
		
		path = "./Models/"
		name = path + "stl_verification_" + TEST_FILE + "_feat" + MODEL_NAME + ".pt"
		
		tweetconfig.output_attentions = False
		tweetconfig.output_hidden_states = False	
		model = TreeLSTM(MODEL_NAME, TRAINABLE_LAYERS, IN_FEATURES, OUT_FEATURES, CLASSIFIER_DROPOUT, mode="cls", tweetconfig=tweetconfig)
		model.cuda(gpu_id)
		# model.cuda()
		tweetconfig2 = copy.deepcopy(tweetconfig)
		tweetconfig2.output_attentions = True
		tweetconfig2.output_hidden_states = True
		test_model = TreeLSTM(MODEL_NAME, TRAINABLE_LAYERS, IN_FEATURES, OUT_FEATURES, CLASSIFIER_DROPOUT, mode="cls", tweetconfig=tweetconfig2)
		test_model.cuda(gpu_id)
		# test_model.cuda()
		
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
		print('\nTraining Set:', set(files) - {test_file})
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
		
		# print("size of training data", sum([len(i) for i in (train_trees)]))
		print("size of training data", len(train_trees))
		print("Size of test data", len(test_trees))		
		print("\ntraining started....")
		
		prev_loss = math.inf
		prev_acc = 0		

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
				shuffle = True)
			
			model.train()

			ground_labels = []
			predicted_labels = []
			j = 0
			train_avg_loss=0					
			err_count = 0
			for tree_batch in data_gen:
				loss, p_labels, g_labels, err = train(tree_batch, test_file, "train")
				err_count += err
				if err != 1:
					ground_labels.extend(g_labels)
					predicted_labels.extend(p_labels)
					j = j+1
					train_avg_loss += loss					
				# torch.cuda.empty_cache()
			train_acc = accuracy_score(ground_labels, predicted_labels)
			train_avg_loss /= j
			
			
			print("validation started..", len(val_data))			
			model.eval()
			
			val_ground_labels = []
			val_predicted_labels= []
			val_j = 0
			val_avg_loss = 0			
			with torch.no_grad():
				for batch in val_gen:
					loss, p_labels, g_labels, err = train(batch, test_file, "eval")
					err_count += err
					if err != 1:
						val_ground_labels.extend(g_labels)
						val_predicted_labels.extend(p_labels)
						val_j += 1
						val_avg_loss += loss
					# torch.cuda.empty_cache()			
			val_acc = accuracy_score(val_ground_labels, val_predicted_labels)
			# val_f1 = f1_score(val_ground_labels, val_predicted_labels)
			val_avg_loss /= val_j
			
			if MODEL_SAVING_POLICY == "acc":
				if(prev_acc <= val_acc):
					save_model(model, name, val_acc, val_avg_loss)
					prev_acc = val_acc
			else:			
				if(prev_loss >= val_avg_loss):
					save_model(model, name, val_acc, val_avg_loss)
					prev_loss = val_avg_loss
			
			print('\nIteration ', i)
			print("errors ", err_count)			
			print('Training Loss: ', train_avg_loss)
			print('Training accuracy: ', train_acc)	
			print('Validation loss: ', val_avg_loss)			
			print('Validation accuracy: ', val_acc)
			# print('Validation f1 score: ', val_f1)
			# print('Training confusion matrix: ', confusion_matrix(ground_labels, predicted_labels))
			
			train_accuracy.append(train_acc)
			val_accuracy.append(val_acc)
			
			# scheduler.step(val_acc)

			if (i+1) % 5 == 0 and i > 0:
				load_model(test_model, name)
				print('Now Testing:', test_file)
				total = 0
				tweet_ids = []
				predicted = []
				prob = []
				ground = []
				token_attentions = []
				# num_tokens = []
				# tokenslist = []

				test_model.eval()
				with torch.no_grad():
					for test in test_trees:
						try:
							h_test, h_test_root, c, att = test_model(
									test['f'].to(device),
									test['a'].to(device),
									test['k'].to(device),
									test['node_order'].to(device),
									test['adjacency_list'].to(device),
									test['edge_order'].to(device),
									test['root_n'].to(device)
									# test['root_l'].to(device)
							)
						except:
							continue
						lastlayer_attention = att[-1][0]
						lastlayer_attention = lastlayer_attention.to("cpu")
						# print("shape of cls:", lastlayer_attention.shape)
						a = torch.mean(lastlayer_attention, dim=0).squeeze(0)
						# print("after mean:", a.shape)
						cls_attentions = a[0]
						token_attentions.append(cls_attentions)
						# tokens = tokenizer.convert_ids_to_tokens(test['f'][0])
						# tokenslist.append(tokens)
						# num_tokens.append(int(torch.sum(test['a'][0]).item()))

						tweet_ids.append(test['tweet_id'].item())

						true_label_val = test['root_l'].to('cpu')					
						true_label = true_label_val[0][1].item()
						# print(true_label)
						pred_logit = h_test_root.detach().cpu()					
						logit_after_sigmoid = sigmoid_fn(pred_logit)
						# print(logit_after_sigmoid)
						pred_label = 1 if logit_after_sigmoid[0].item() >= 0.5 else 0
						pred_prob = logit_after_sigmoid[0].item() if pred_label == 1 else 1 - logit_after_sigmoid[0].item()
						# print(pred_label)
						
						# true_label_val = test['root_l'].to('cpu')
						# true_label = true_label_val[0][1]
						# pred_logit = h_test_root.detach().cpu()
						# For getting the predicted label, it's immaterial whether we apply softmax or not
						# pred_v, pred_label = torch.max(pred_logit, 1)
						# pred_v, pred_label = torch.max(F.softmax(pred_logit, dim=1), 1)						
						
						predicted.append(pred_label)
						prob.append(pred_prob)
						ground.append(true_label)
						
						total += 1
				
				print(f'\nTotal Test trees evaluated: {total}')
				accuracy = accuracy_score(ground, predicted)
				print('Accuracy: %f' % accuracy)
				# precision tp / (tp + fp)
				precision = precision_score(ground, predicted)
				print('Precision: %f' % precision)
				precision = precision_score(ground, predicted, average='macro')
				print('Macro Precision: %f' % precision)				
				# recall: tp / (tp + fn)
				recall = recall_score(ground, predicted)
				print('Recall: %f' % recall)
				recall = recall_score(ground, predicted, average='macro')
				print('Macro Recall: %f' % recall)
				# f1: 2 tp / (2 tp + fp + fn)
				f1 = f1_score(ground, predicted)
				print('F1 score: %f' % f1)
				f1 = f1_score(ground, predicted, average='macro')
				print('Macro F1 score: %f' % f1)
				print("\n\n")
				print(classification_report(ground, predicted, digits=5))
				print("\n\n")
				print('confusion matrix ', confusion_matrix(ground, predicted))

				# df = pd.DataFrame({
				# 				"Tweet_ID": tweet_ids, 
				# 				"pred": predicted, 
				# 				"pred_prob": prob, 
				# 				"gt": ground
				# 				# "Tokens": tokenslist, 
				# 				"Attentions": token_attentions}
				# 				# "Numtokens": num_tokens}
				# 			)
				# if L2_REGULARIZER == 'n':
				# 	df.to_pickle("STL_" + TEST_FILE + "_"  + MODEL_NAME + "_" + str(IN_FEATURES) + "_L2_n_LR_" + str(lr) + "_LOSS_" + LOSS_FN + ".pkl")
				# else:
				# 	df.to_pickle("STL_" + TEST_FILE + "_"  + MODEL_NAME + "_" + str(IN_FEATURES) + "_L2_y_" + str(WEIGHT_DECAY) + "_LR_" + str(lr) + "_LOSS_" + LOSS_FN + ".pkl")

		
		# plt.plot(numpy.array(train_accuracy))
		# plt.plot(numpy.array(val_accuracy))
		# plt.legend(['train_acc','val_acc'])
		# plt.show()
		# print('Iteration ', i+1,' Loss: ', total_loss)
		print('Training Complete')

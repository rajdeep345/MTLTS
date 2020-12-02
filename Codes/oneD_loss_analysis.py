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
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import Dataset, IterableDataset, DataLoader, TensorDataset
from torch.utils.data import random_split, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import *

seed_val = 1955
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)
random.seed(seed_val)
numpy.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)

device = 'cuda'
USE_DROPOUT = 'n'

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


def convert_tree_to_tensors(tree, device=device):
	# Label each node with its walk order to match nodes to feature tensor indexes
	# This modifies the original tree as a side effect
	_label_node_index(tree)

	features = _gather_node_attributes(tree, 'f')
	attention = _gather_node_attributes(tree, 'a')
	# old_features = _gather_node_attributes(tree, 'k')
	labels = _gather_node_attributes(tree, 'l')		
	root_label = labels[0][1]
	adjacency_list = _gather_adjacency_list(tree)
	summ_gt = tree['summ_gt_clean']

	node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(features))
	root_node = [0]

	return {
		'f': torch.tensor(features, dtype=torch.long),
		'a':torch.tensor(attention,  dtype=torch.float32),
		'root_l': torch.tensor(root_label, dtype=torch.long),
		'root_n': torch.tensor(root_node,  dtype=torch.int64),
		'node_order': torch.tensor(node_order,  dtype=torch.int64),
		'adjacency_list': torch.tensor(adjacency_list,  dtype=torch.int64),
		'edge_order': torch.tensor(edge_order,  dtype=torch.int64),
		'summ_gt': torch.tensor(summ_gt, dtype=torch.int64)
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
	# batched_old_features = torch.cat([b['k'] for b in batch])
	batched_node_order = torch.cat([b['node_order'] for b in batch])

	idx = 0
	root_li = []

	for b in batch:
		root_li.append(idx)
		idx += len(b['node_order'])

	batched_root = torch.tensor(root_li, dtype=torch.int64)

	batched_edge_order = torch.cat([b['edge_order'] for b in batch])

	# batched_labels = torch.cat([b['l'] for b in batch])

	# batched_root_labels = torch.cat([b['root_l'] for b in batch])
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
		# 'k': batched_old_features,
		'node_order': batched_node_order,
		'edge_order': batched_edge_order,
		'adjacency_list': batched_adjacency_list,
		'tree_sizes': tree_sizes,
		'root_node': batched_root,
		'root_label': batched_root_labels,
		'summ_gt': batched_summ_labels,
		# 'root_label': batched_root_labels,
		# 'l': batched_labels
	}


def unbatch_tree_tensor(tensor, tree_sizes):
	'''Convenience functo to unbatch a batched tree tensor into individual tensors given an array of tree_sizes.

	sum(tree_sizes) must equal the size of tensor's zeroth dimension.
	'''
	return torch.split(tensor, tree_sizes, dim=0)

MODEL_NAME = "BERT"
TRAINABLE_LAYERS = [0,1,2,3,4,5,6,7,8,9,10,11]
IN_FEATURES = 768
OUT_FEATURES = 128
CLASSIFIER_DROPOUT = 0.4
PLACE = "germanwings-crash.txt"

def load_model(model, path):
	state = torch.load(path)
	model.load_state_dict(state['model'])
	print('Validation accuracy of the model is ', state.get('val_acc'))
	print('Validation loss of the model is ', state.get('val_loss'))
	return state.get('val_acc')

tree_path = './data/PT_PHEME5_FeatBERT40_Depth5_maxR5_MTL_Final/'
files = ['charliehebdo.txt', 'ottawashooting.txt', 'sydneysiege.txt','germanwings-crash.txt']
data = []
for filename in files:
	if filename==PLACE:
		continue
	input_file = codecs.open(tree_path + filename, 'r', 'utf-8')
	for row in input_file:
		s = row.strip().split('\t')		
		tweet_id = int(s[0])
		curr_tree = eval(s[1])
		data.append(curr_tree)

tree_li = []
for curr_tree in data:
    curr_tensor = convert_tree_to_tensors(curr_tree)
    tree_li.append(curr_tensor)
BATCH_SIZE = 256
data_gen = DataLoader(
                tree_li,
                collate_fn=batch_tree_input,
                batch_size=BATCH_SIZE,
                shuffle = False)

"""# MTL"""

class TreeLSTM(torch.nn.Module):
	'''PyTorch TreeLSTM model that implements efficient batching.
	'''
	def __init__(self, model_name, trainable_layers, in_features, out_features, classifier_dropout, mode, tweetconfig):
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
			self.BERT_model = RobertaModel.from_pretrained("./MTL_VeriSumm/BERTweet_base_transformers/model.bin", config=tweetconfig)

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

def compute_weights(alpha):
    W1 = model1.state_dict()
    W2 = model2.state_dict()
    W3 = test_model.state_dict()
    for key in W1:
        W3[key] = W1[key] + alpha*(W2[key] - W1[key])
    test_model.load_state_dict(W3)

model1 = TreeLSTM(MODEL_NAME, TRAINABLE_LAYERS, IN_FEATURES, OUT_FEATURES, CLASSIFIER_DROPOUT, mode="cls", tweetconfig=None)
# path to trained MTL model
mtl_path = "./data/stl_veri/mtl_german.pt"
model2 = TreeLSTM(MODEL_NAME, TRAINABLE_LAYERS, IN_FEATURES, OUT_FEATURES, CLASSIFIER_DROPOUT, mode="cls", tweetconfig=None)
load_model(model2, mtl_path)
test_model = TreeLSTM(MODEL_NAME, TRAINABLE_LAYERS, IN_FEATURES, OUT_FEATURES, CLASSIFIER_DROPOUT, mode="cls", tweetconfig=None)
test_model.cuda()

def train():
	test_model.eval()
	veri_l = []
	summ_l = []
	with torch.no_grad():
		for batch in tqdm(data_gen):
			try:
				h_test, h_test_root, c, summ_out, att= test_model(
						batch['f'].to(device),
						batch['a'].to(device),
						None,
						batch['node_order'].to(device),
						batch['adjacency_list'].to(device),
						batch['edge_order'].to(device),
						batch['root_node'].to(device)
						# test['root_l'].to(device)
				)
			except Exception as e:
				print(e)
				continue
			
			pred_logits = h_test_root.detach().cpu()
			prob = F.softmax(pred_logits, dim=1)
			pred_v, pred_label = torch.max(F.softmax(pred_logits, dim=1), 1)

			loss_function_veri = CrossEntropyLoss()
			loss_function_summ = CrossEntropyLoss()
			loss_summ = loss_function_summ(summ_out, batch['summ_gt'].long().to(device))

			loss_veri = loss_function_veri(h_test_root, batch['root_label'].long().to(device))
			veri_l.append(loss_veri.item())
			summ_l.append(loss_summ.item())

	return  sum(veri_l)/len(veri_l), sum(summ_l)/len(summ_l)


veri_loss = []
summ_loss = []
for alpha in numpy.arange(-4,4.25,0.25):
    compute_weights(alpha)
    loss = train()
    veri_loss.append(loss[0])
    summ_loss.append(loss[1])

print("veri loss", veri_loss)
print("summ loss",summ_loss)



""" HMTL Model """

class Hierarchial_MTL(torch.nn.Module):
	'''PyTorch Hierarchial_MTL model that implements efficient batching.
	'''
	def __init__(self, model_name, trainable_layers, in_features, out_features, classifier_dropout, mode, tweetconfig=None):
		'''Hierarchial_MTL class initializer

		Takes in int sizes of in_features and out_features and sets up model Linear network layers.
		'''
		super().__init__()
		print("model intialising...")
		self.in_features = in_features
		self.out_features = out_features
		self.mode = mode
		
		if model_name == 'BERT':
			if AB == 0:
				self.BERT_model = BertModel.from_pretrained("bert-base-cased", output_attentions=True)
			else:
				self.BERT_model = BertModel(output_attentions=True)
		elif model_name == 'ROBERTA':
			self.BERT_model = RobertaModel.from_pretrained("roberta-base", output_attentions=True)
		elif model_name == 'BERTWEET':
			self.BERT_model = RobertaModel.from_pretrained("/home/rajdeep/MTL_VeriSumm/BERTweet_base_transformers/model.bin", config=tweetconfig)
		

		self.W_iou = torch.nn.Linear(self.in_features, 3 * self.out_features)
		self.U_iou = torch.nn.Linear(self.out_features, 3 * self.out_features, bias=False)

		# f terms are maintained seperate from the iou terms because they involve sums over child nodes
		# while the iou terms do not
		self.W_f = torch.nn.Linear(self.in_features, self.out_features)
		self.U_f = torch.nn.Linear(self.out_features, self.out_features, bias=False)
		
		# self.veri_fc = torch.nn.Linear(self.out_features, 1)
		self.veri_fc = torch.nn.Linear(self.out_features, 2)
		
		# self.bert_dropout = torch.nn.Dropout(bert_dropout)		
		self.classifier_dropout = torch.nn.Dropout(classifier_dropout)		
		
		# Summarizer layers.
		# self.summ_fc1 = torch.nn.Linear(self.in_features + 130 , self.out_features)
		self.summ_fc1 = torch.nn.Linear(self.in_features + 2 , self.out_features)
		self.summ_fc2 = torch.nn.Linear(self.out_features, 2)


	
	def forward(self, features, attentions, node_order, adjacency_list, edge_order, root_node, trainable_part=None):
		'''Run Hierarchial_MTL model on a tree data structure with node features

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

		if trainable_part == 'verifier':
			return self.verify_forward(h, c, output_vectors, node_order, adjacency_list, edge_order, root_node)
		elif trainable_part == 'summarizer':
			h_root, veri_logits_out = self.verify_forward(h, c, output_vectors, node_order, adjacency_list, edge_order, root_node)
			return veri_logits_out, self.summarize_forward(h_root, veri_logits_out, output_vectors, root_node), att
				

	def verify_forward(self, h, c, output_vectors, node_order, adjacency_list, edge_order, root_node):
		# Verification
		for n in range(node_order.max() + 1):
			self._run_lstm(n, h, c, output_vectors, node_order, adjacency_list, edge_order)

		h_root = h[root_node, :]
		if USE_DROPOUT == 'y':
			h_root = self.classifier_dropout(h_root)
		
		veri_logits_out = self.veri_fc(h_root)
		return h_root, veri_logits_out

	def summarize_forward(self, h_root, veri_logits_out, output_vectors, root_node):
		# Summarization
		# summ_in = output_vectors[root_node,:]
		h_root_dash = h_root.detach().clone().requires_grad_(False)
		veri_logits_out_dash = veri_logits_out.detach().clone().requires_grad_(False)
		# output_vectors_dash = output_vectors.detach().clone().requires_grad_(False)
		
		# Use following lines to allow connection between summarizer and verifier
		# summ_in = torch.cat([output_vectors[root_node,:], h_root, F.softmax(veri_logits_out, dim=1)], axis=1)
		summ_in = torch.cat([output_vectors[root_node,:], F.softmax(veri_logits_out, dim=1)], axis=1)

		# Use following lines to disconnect summarizer and verifier
		# summ_in = torch.cat([output_vectors[root_node, :], h_root_dash, F.softmax(veri_logits_out_dash, dim=1)], axis=1)
		# summ_in = torch.cat([output_vectors[root_node, :], F.softmax(veri_logits_out_dash, dim=1)], axis=1)
		
		if USE_DROPOUT == 'y':
			summ_in = self.classifier_dropout(summ_in)
		summ_logits1 = self.summ_fc1(summ_in)
		summ_logits_out = self.summ_fc2(summ_logits1)
		return summ_logits_out


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

def compute_weights(alpha):
    W1 = model1_hmtl.state_dict()
    W2 = model2_hmtl.state_dict()
    W3 = test_model_hmtl.state_dict()
    for key in W1:
        W3[key] = W1[key] + alpha*(W2[key] - W1[key])
    test_model_hmtl.load_state_dict(W3)

def train_loop(model:Hierarchial_MTL, theta_dash=None):
    sloss = []
    vloss = []
    model.eval()
    module = 'summarizer'
    with torch.no_grad():
        for tree_batch in tqdm(data_gen):
            outputs = model(
                tree_batch['f'].to(device),
                tree_batch['a'].to(device),
                # tree_batch['k'].to(device),
                tree_batch['node_order'].to(device),
                tree_batch['adjacency_list'].to(device),
                tree_batch['edge_order'].to(device),
                tree_batch['root_node'].to(device),
                trainable_part=module
            )
            
            summ_out = outputs[1]
            summ_gt_labels = tree_batch['summ_gt'].to('cpu')	
            summ_gt_labels_tensor = summ_gt_labels.clone().detach().type_as(summ_out).to(device)
            summ_logits = summ_out.detach().cpu()
            pred_v_summ, pred_labels_summ = torch.max(F.softmax(summ_logits, dim=1), 1)
            loss_function_summ = CrossEntropyLoss()
            
            veri_logits_out = outputs[0]
            g_labels = tree_batch['root_label'].to('cpu')
            g_labels_tensor = g_labels.clone().detach().type_as(veri_logits_out).to(device)
            pred_logits = veri_logits_out.detach().cpu()
            pred_v, pred_labels = torch.max(F.softmax(pred_logits, dim=1), 1)
            loss_function_veri = CrossEntropyLoss()

            veri_loss = loss_function_veri(veri_logits_out, g_labels_tensor.long())
            summ_loss = loss_function_summ(summ_out, tree_batch['summ_gt'].long().to(device))
            vloss.append(veri_loss.item())
            sloss.append(summ_loss.item())
    return sum(vloss)/len(vloss), sum(sloss)/len(sloss)

AB = 0
model1_hmtl = Hierarchial_MTL(MODEL_NAME, TRAINABLE_LAYERS, IN_FEATURES, OUT_FEATURES, CLASSIFIER_DROPOUT, mode="cls", tweetconfig=None)
# path to HMTL trained model
hmtl_path = "./data/HMTL_weights/hmtl_german_2e-5.pt"
model2_hmtl = Hierarchial_MTL(MODEL_NAME, TRAINABLE_LAYERS, IN_FEATURES, OUT_FEATURES, CLASSIFIER_DROPOUT, mode="cls", tweetconfig=None)
load_model(model2_hmtl, hmtl_path)
test_model_hmtl = Hierarchial_MTL(MODEL_NAME, TRAINABLE_LAYERS, IN_FEATURES, OUT_FEATURES, CLASSIFIER_DROPOUT, mode="cls", tweetconfig=None)
test_model_hmtl.cuda()

from tqdm import tqdm
veri_loss_hmtl = []
summ_loss_hmtl = []
for alpha in numpy.arange(-4,4.25,0.25):
    compute_weights(alpha)
    loss = train_loop(test_model_hmtl,theta_dash=None)
    veri_loss_hmtl.append(loss[0])
    summ_loss_hmtl.append(loss[1])

"""## 1D Loss Plot"""

x = list(numpy.arange(-4,4.25,0.25))
import matplotlib.pyplot as plt
plt.plot(x,veri_loss, label = 'MTL verification loss')
plt.plot(x,veri_loss_hmtl,'r',ls='--',label='HMTL veirfication loss')
plt.ylabel('Average Training Loss')
plt.xlabel('alpha')
plt.legend(loc='upper left')
# plt.show()
plt.savefig('mtl_hmtl_german_analysis.png')
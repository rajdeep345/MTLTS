import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import numpy
import os
import codecs
import random
import sys
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from transformers import *
from tqdm import tqdm
import itertools

# # If there's a GPU available...
if torch.cuda.is_available():
	# Tell PyTorch to use the GPU.    
	device = torch.device("cuda")
	print('There are %d GPU(s) available.' % torch.cuda.device_count())
	print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
	print('No GPU available, using the CPU instead.')
	device = torch.device("cpu")


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
	root_label = [labels[0]]
	adjacency_list = _gather_adjacency_list(tree)

	node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(features))
	root_node = [0]

	return {
		'f': torch.tensor(features, dtype=torch.long),
		'a':torch.tensor(attention,  dtype=torch.float32),
		# 'k':torch.tensor(old_features, dtype=torch.float32),
		'l': torch.tensor(labels,  dtype=torch.float32),
		'root_l': torch.tensor(root_label, dtype=torch.float32),
		'root_n': torch.tensor(root_node,  dtype=torch.int64),
		'node_order': torch.tensor(node_order,  dtype=torch.int64),
		'adjacency_list': torch.tensor(adjacency_list,  dtype=torch.int64),
		'edge_order': torch.tensor(edge_order,  dtype=torch.int64),
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

	batched_root = torch.tensor(root_li,dtype=torch.int64)

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
		# 'k': batched_old_features,
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
	def __init__(self, in_features, out_features,mode):
		'''TreeLSTM class initializer

		Takes in int sizes of in_features and out_features and sets up model Linear network layers.
		'''
		super().__init__()
		print("model intialising...")
		self.in_features = in_features
		self.out_features = out_features
		self.mode = mode
		self.BERT_model  = BertModel.from_pretrained("bert-base-cased")
				
		self.W_iou = torch.nn.Linear(self.in_features, 3 * self.out_features)
		self.U_iou = torch.nn.Linear(self.out_features, 3 * self.out_features, bias=False)

		# f terms are maintained seperate from the iou terms because they involve sums over child nodes
		# while the iou terms do not
		self.W_f = torch.nn.Linear(self.in_features, self.out_features)
		self.U_f = torch.nn.Linear(self.out_features, self.out_features, bias=False)
		self.fc = torch.nn.Linear(self.out_features, 2)
	
	
	# def forward(self, features,attentions,old_features,node_order, adjacency_list, edge_order, root_node, root_label):
	def forward(self, features, attentions, node_order, adjacency_list, edge_order, root_node, root_label):		
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
		hidden_states,_ = self.BERT_model(input_ids=features,attention_mask=attentions)

		if self.mode=="cls":
			output_vectors = hidden_states[:,0]
			# output_vectors = torch.cat([output_vectors, old_features], axis=1)
		if self.mode=="avg":
			input_mask_expanded = attentions.unsqueeze(-1).expand(hidden_states.size()).float()
			sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
			sum_mask = input_mask_expanded.sum(1)
			output_vectors= sum_embeddings / sum_mask
			# output_vectors = torch.cat([output_vectors, old_features], axis=1)
		
		for n in range(node_order.max() + 1):
			self._run_lstm(n, h, c, output_vectors, node_order, adjacency_list, edge_order)

		h_root = h[root_node, :]
		out = self.fc(h_root)
		out = torch.nn.functional.softmax(out, dim = 1)
		return h, out, c

	
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


def train(tree_batch, mode="train"):
	err_count = 0
	loss = 0
	pred_label = []
	g_labels = []
	
	# try:
	h, h_root, c = model(
		tree_batch['f'].to(device),
		tree_batch['a'].to(device),
		# tree_batch['k'].to(device),
		tree_batch['node_order'].to(device),
		tree_batch['adjacency_list'].to(device),
		tree_batch['edge_order'].to(device),
		tree_batch['root_node'].to(device),
		tree_batch['root_label'].to(device)
	)

	root_labels = tree_batch['root_label'].to(device)
	pred_label_vals = h_root.detach().cpu()
	pred_v, pred_label = torch.max(pred_label_vals, 1)
	root = root_labels.to('cpu')
	g_labels = [t[1] for t in root]
	loss = loss_function(h_root, root_labels)
	optimizer.zero_grad()
	
	if mode == "train":
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()
	# except Exception as e:
	#     print("here error2 ",e)
	#     err_count = 1
	
	return loss, pred_label, g_labels, err_count


tree_path = './Parsed-Trees-Pad32_FeatBERT_Depth5_maxR5/'
files = ['charliehebdo.txt', 'germanwings-crash.txt', 'ottawashooting.txt','sydneysiege.txt']

tree_li = {}
val_li = {}
for filename in files:
	input_file = codecs.open(tree_path + filename, 'r', 'utf-8')
	tree_li[filename]=[]
	for row in input_file:
		s = row.strip().split('\t')
		tweet_id = s[0]
		curr_tree = eval(s[1])
		# try:
		curr_tensor = convert_tree_to_tensors(curr_tree)
		# except Exception as e:
		#     # print(e)
		#     continue

		tree_li[filename].append(curr_tensor)
		# tree_li.append(curr_tree)
	random.shuffle(tree_li[filename])
	val_len = int(0.1*len(tree_li[filename]))
	val_li[filename] = (tree_li[filename][:val_len])
	tree_li[filename] = tree_li[filename][val_len:] 
	input_file.close()
	print(filename, len(tree_li[filename]))



files = ['charliehebdo.txt', 'ottawashooting.txt', 'germanwings-crash.txt','sydneysiege.txt']
lr_list = [1e-5, 2e-5]
for lr in lr_list:
	print("\n\n\nTraining with LR: ", lr)
	for test in files:
		seed_val = 40
		random.seed(seed_val)
		numpy.random.seed(seed_val)
		torch.manual_seed(seed_val)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

		path = "./Models/"
		IN_FEATURES = 768
		OUT_FEATURES = 128
		NUM_ITERATIONS = 10
		BATCH_SIZE = 16
		name = path + "stl_verification_featBERT.pt"
		model = TreeLSTM(IN_FEATURES, OUT_FEATURES, mode="cls").train()
		model.cuda()
		test_model = TreeLSTM(IN_FEATURES, OUT_FEATURES, mode="cls")
		test_model.cuda()
		loss_function = torch.nn.BCEWithLogitsLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
		
		print("Size of test data", len(test_trees))
		print("size of training data", sum([len(i) for i in (train_trees)]))
		print("\ntraining started....")
		prev_loss = 1
		prev_acc = 0		
		for i in range(NUM_ITERATIONS):
			model.train()
			# model.zero_grad() 
			total_loss = 0
			length = []
			data_gen = DataLoader(
				train_trees,
				collate_fn=batch_tree_input,
				batch_size=BATCH_SIZE,
				shuffle = True
			)

			val_gen = DataLoader(val_data,
					collate_fn=batch_tree_input,
					batch_size=BATCH_SIZE,
					shuffle = True)
			
			j = 0
			avg_loss=0
			ground_labels = []
			predicted_labels = []
			val_ground_labels = []
			val_predicted_labels= []
			err_count = 0
			for tree_batch in data_gen:
				loss,p_labels,g_labels,err = train(tree_batch,"train")
				err_count+=err
				if err!=1:
					ground_labels.extend(g_labels)
					predicted_labels.extend(p_labels)
					j = j+1
					avg_loss += loss
					total_loss += loss
				# torch.cuda.empty_cache() 
			
			print("validation started..",len(val_data))
			model.eval()
			val_avg_loss = 0
			val_j = 0
			with torch.no_grad():
				for batch in val_gen:
					loss,p_labels,g_labels,err = train(batch,"eval")
					err_count+=err
					if err!=1:
						val_ground_labels.extend(g_labels)
						val_predicted_labels.extend(p_labels)
						val_j += 1
						val_avg_loss += loss
					# torch.cuda.empty_cache()
			val_acc = accuracy_score(val_ground_labels,val_predicted_labels)
			val_f1 = f1_score(val_ground_labels,val_predicted_labels)
			val_loss = val_avg_loss/val_j
			'''
			if(prev_acc<=val_acc):
				save_model(model, name, val_acc, val_loss)
				prev_acc = val_acc
			'''        
			if(prev_loss>=val_loss):
				save_model(model, name, val_acc, val_loss)
				prev_loss = val_loss
			
			print("errors ",err_count)
			print('Iteration ', i)
			print('Training Loss: ', avg_loss/j)	
			print('Validation loss: ', val_loss)
			print('training accuracy: ', accuracy_score(ground_labels,predicted_labels))
			print('Validation accuracy: ', val_acc)
			print('Validation f1 score: ',val_f1)
			print('Training confusion matrix: ', confusion_matrix(ground_labels, predicted_labels))

			if ((i+1) % 5 == 0 and i > 0):
				load_model(test_model, name)
				print('Now Testing:', test_file)
				acc = 0
				total = 0
				predicted = []
				ground = []
				test_model.eval()
				with torch.no_grad():
					for test in test_trees:
						try:
							h_test,h_test_root,c = test_model(
									test['f'].to(device),
									test['a'].to(device),
									# test['k'].to(device),
									test['node_order'].to(device),
									test['adjacency_list'].to(device),
									test['edge_order'].to(device),
									test['root_n'].to(device),
									test['root_l'].to(device)
							)
						except:
							continue
						true_label_vals = test['root_l'].to('cpu')
						pred_label_vals = h_test_root.detach().cpu()
						pred_v, pred_label = torch.max(pred_label_vals, 1)

						true_label = true_label_vals[0][1]
						predicted.append(pred_label)
						ground.append(true_label)
						if pred_label == true_label:
							acc += 1
						total += 1
				print(test_file, 'accuracy:', acc / total)
				accuracy = accuracy_score(ground,predicted)

				print('Accuracy: %f' % accuracy)
				# precision tp / (tp + fp)
				precision = precision_score(ground,predicted)
				print('Precision: %f' % precision)
				# recall: tp / (tp + fn)
				recall = recall_score(ground,predicted)
				print('Recall: %f' % recall)
				# f1: 2 tp / (2 tp + fp + fn)
				f1 = f1_score(ground,predicted)
				print('F1 score: %f' % f1)
				f1 = f1_score(ground,predicted, average='macro')
				print('Macro F1 score: %f' % f1)
				print('confusion matrix ', confusion_matrix(ground, predicted))            
		
		print('Iteration ', i+1,' Loss: ',total_loss)
		print('Training Complete')

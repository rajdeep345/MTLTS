import codecs
import random
import pandas as pd
import numpy as np
import torch
from TreeData import convert_tree_to_tensors
from sklearn.utils.class_weight import compute_class_weight

# Manual SIT Labels
# dfc = pd.read_pickle("data/manual_sit/dfc_0625.pkl")
# dfg = pd.read_pickle("data/manual_sit/dfg_08.pkl")
# dfo = pd.read_pickle("data/manual_sit/dfo_069.pkl")
# dfs = pd.read_pickle("data/manual_sit/dfs_0675.pkl")

# Model SIT Labels
path = "./drive/My Drive/Data/"
dfc = pd.read_pickle(path+"dfc_0625.pkl")
dfg = pd.read_pickle(path+"dfg_08.pkl")
dfo = pd.read_pickle(path+"dfo_069.pkl")
dfs = pd.read_pickle(path+"dfs_0675.pkl")


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


def read_data(tree_path, files, device):
	tree_li = {}
	train_li = {}
	val_li = {}
	for filename in files:
		input_file = codecs.open(tree_path + filename, 'r', 'utf-8')
		tree_li[filename]=[]
		for row in input_file:
			s = row.strip().split('\t')		
			tweet_id = int(s[0])
			curr_tree = eval(s[1])
			curr_tensor = convert_tree_to_tensors(curr_tree, tweet_id, device)
			tree_li[filename].append(curr_tensor)
			
		random.shuffle(tree_li[filename])
		train_li[filename], val_li[filename] = split_data(tree_li[filename], 0.1)
		input_file.close()
		print(f'{filename} Training Set Size: {len(train_li[filename])}, Validation Set Size: {len(val_li[filename])}, Total: {len(tree_li[filename]) + len(val_li[filename])}')
	return train_li, val_li


def compute_weights(train_li, files, device):
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
				for tree in train_li[filename]:
					# print(int(tree['root_l'].tolist()[0][1]))
					y.append(int(tree['root_l'].tolist()))
					summ_y.append(int(tree['summ_gt'].tolist()))
					file_dist[int(tree['root_l'].tolist())] += 1
					summ_file_dist[int(tree['summ_gt'].tolist())] += 1
					label_dist[int(tree['root_l'].tolist())] += 1
					summ_label_dist[int(tree['summ_gt'].tolist())] += 1
				
		print(f'Total non-rumors: {label_dist[0]}, Total rumors: {label_dist[1]}')
		print(f'Total non-summary-tweets: {summ_label_dist[0]}, Total summry-tweets: {summ_label_dist[1]}')
		weight_vec[test_file] = torch.tensor(compute_class_weight('balanced', np.unique(y), y)).to(device)
		summ_weight_vec[test_file] = torch.tensor(compute_class_weight('balanced', np.unique(summ_y), summ_y)).to(device)
		pos_weight = label_dist[0] / label_dist[1]
		pos_weight_vec[test_file] = torch.tensor([pos_weight], dtype=torch.float32).to(device)
		summ_pos_weight = summ_label_dist[0] / summ_label_dist[1]
		summ_pos_weight_vec[test_file] = torch.tensor([summ_pos_weight], dtype=torch.float32).to(device)
		print(f'Test File: {test_file}, Verification Weight Vector: {weight_vec[test_file]}')
		print(f'Test File: {test_file}, Verification Pos Weight Vector: {pos_weight_vec[test_file]}')
		print(f'Test File: {test_file}, Summary Weight Vector: {summ_weight_vec[test_file]}')
		print(f'Test File: {test_file}, Summary Pos Weight Vector: {summ_pos_weight_vec[test_file]}')
	return weight_vec, summ_weight_vec, pos_weight_vec, summ_pos_weight_vec
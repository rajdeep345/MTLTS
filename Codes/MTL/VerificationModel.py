import torch
from torch import nn
from TreeLSTM import *
import torch.nn.functional as F
import numpy as np


class VerificationModel(torch.nn.Module):
	'''PyTorch TreeLSTM model that implements efficient batching.
	'''
	def __init__(self, args, in_features, out_features, classifier_dropout):
		'''TreeLSTM class initializer

		Takes in int sizes of in_features and out_features and sets up model Linear network layers.
		'''
		super().__init__()
		print("model intialising...")
		self.args = args
		self.out_features = out_features

		self.tree_lstm = TreeLSTM(args, in_features,out_features)
		if args.use_situ_tag == 'y':			
			self.fc = torch.nn.Linear(self.out_features+1, 2)
		else:
			# self.fc = torch.nn.Linear(self.out_features, 1)
			self.fc = torch.nn.Linear(self.out_features, 2)
			
		self.classifier_dropout = torch.nn.Dropout(classifier_dropout)

	
	def forward(self, inputs, sit_tags, node_order, adjacency_list, edge_order, root_node):

		h_root = self.tree_lstm(inputs = inputs, 
                                sit_tags = sit_tags, 
                                node_order = node_order, 
                                adjacency_list = adjacency_list, 
                                edge_order = edge_order, 
                                root_node = root_node)
		if self.args.use_situ_tag == 'y':
			sit_tags = sit_tags.view(-1,1)
			h_root = torch.cat([h_root, sit_tags], axis=1)
		
		if self.args.use_dropout == 'y':
			h_root = self.classifier_dropout(h_root)
		
		veri_logits_out = self.fc(h_root)
		
		return h_root, veri_logits_out


	
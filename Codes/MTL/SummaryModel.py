import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class SummaryModel(torch.nn.Module):
	'''PyTorch TreeLSTM model that implements efficient batching.
	'''
	def __init__(self, args, in_features, out_features):
		'''TreeLSTM class initializer

		Takes in int sizes of in_features and out_features and sets up model Linear network layers.
		'''
		super().__init__()
		print("model intialising...")
		self.args = args
		self.in_features = in_features
		self.out_features = out_features
		
		if args.use_situ_tag == 'y':
			self.summ_fc1 = torch.nn.Linear(self.in_features+1, self.out_features)
		else:
			self.summ_fc1 = torch.nn.Linear(self.in_features, self.out_features)
		self.summ_fc2 = torch.nn.Linear(self.out_features, 2)

	

	def forward(self, summ_inputs, sit_tags):
		'''Run TreeLSTM model on a tree data structure with node features

		Takes Tensors encoding node features, a tree node adjacency_list, and the order in which 
		the tree processing should proceed in node_order and edge_order.
		'''

		if self.args.use_situ_tag == 'y':
			sit_tags = sit_tags.view(-1,1)
			summ_in = torch.cat([summ_inputs, sit_tags], axis=1)
		summ_logits1 = self.summ_fc1(summ_inputs)
		summ_logits_out = self.summ_fc2(summ_logits1)
		
		# summ_pred_out = F.softmax(summ_logits_out, dim = 1)
		
		return summ_logits_out



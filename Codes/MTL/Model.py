import os
import torch
import rouge
import subprocess
import numpy as np
import pandas as pd

from Utils import *
from VerificationModel import VerificationModel
from SummaryModel import SummaryModel

from torch import nn
from transformers import *
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, classification_report

class MTLModel(torch.nn.Module):
	'''
	This class has methods for initialising an MTL model for training verification and extracive summarization tasks of disaster related tweets
	'''
	def __init__(self, args, trainable_layers, in_features, out_features, classifier_dropout, mode='cls'):
		'''
		This is the constructer class for MTLModel
		Inputs:
		args(type: object of class argparse.Namespace) - contains hyperparameters set by the user
		trainable_layers(type: list of int) - list of layers which are kept trainable in bert model
		in_features(type:int) - input dimension for tree lstm( dimension of the word embeddings )
		out_features(type:int) - hidden layer size of the model
		mode - (type:str) - ['cls','avg'] - aggregation type for word embeddings
		'''
		super().__init__()
		print("model intialising...")
		self.args = args
		self.in_features = in_features
		self.out_features = out_features
		self.mode = mode
		
		if args.model_name == 'BERT':
			self.BERT_model = BertModel.from_pretrained("bert-base-cased",output_attentions=True)
		elif args.model_name == 'ROBERTA':
			self.BERT_model = RobertaModel.from_pretrained("roberta-base",output_attentions=True)		
		self.veri_model = VerificationModel(args, in_features, out_features, classifier_dropout)	
		self.summ_model = SummaryModel(args, in_features, out_features)


	def forward(self, features, attentions, node_order, adjacency_list, edge_order, root_node, sit_tags):
		"""
		forward function of MTLModel
		Inputs : 
		features(type:tensor of list of ints) - list of tokens sent as input to bert , shape (No of tweets in Batch_size of trees, 32)
		attentions(type:tensor of list of float) - mask of tokens sent as input to bert(0 for padding else 1), shape - (No of tweets in Batch_size of trees, 32)
		node_order, adjacency_list, edge_order, root_node - inputs to tree lstm 
		
		Outputs :
		veri_logits_out - output of the verification model
		summ_logits_out - output of the summarization model
		att - attention for each token in each layer of the bert
		"""
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
		summ_logits_out = self.summ_model(summ_in, sit_tags)
		
		_, veri_logits_out = self.veri_model( inputs = output_vectors, 
						node_order = node_order, 
						adjacency_list = adjacency_list, 
						edge_order = edge_order, 
						root_node = root_node,
						sit_tags = sit_tags)
		
		
		return veri_logits_out, summ_logits_out, att


def run(args, model, optimizer, tree_batch, test_file, device, weight_vec=None, pos_weight_vec=None, summ_weight_vec=None, summ_pos_weight_vec=None, mode="train"):
	"""
	"""
	loss = 0
	h_root, summ_out, att = model(
		tree_batch['f'].to(device),
		tree_batch['a'].to(device),
		# tree_batch['k'].to(device),
		tree_batch['node_order'].to(device),
		tree_batch['adjacency_list'].to(device),
		tree_batch['edge_order'].to(device),
		tree_batch['root_node'].to(device),
		tree_batch['sit_tag'].to(device)
	)

	pred_logits = h_root.detach().cpu()
	pred_v, pred_labels = torch.max(F.softmax(pred_logits, dim=1), 1)
	
	summ_logits = summ_out.detach().cpu()
	pred_v_summ, pred_labels_summ = torch.max(F.softmax(summ_logits, dim=1), 1)
	
	assert mode in ['train','val']

	# if mode!='test':
	weights = weight_vec[test_file]
	pos_weights = pos_weight_vec[test_file]
	summ_weights = summ_weight_vec[test_file]
	summ_pos_weights = summ_pos_weight_vec[test_file]

	g_labels = tree_batch['root_label'].to('cpu')
	g_labels_tensor = g_labels.clone().detach().to(device)
	if args.veri_loss_fn == 'nw':
		loss_function_veri = CrossEntropyLoss()
	else:
		class_weights = weights.float()
		loss_function_veri = CrossEntropyLoss(weight=class_weights)
	loss_veri = loss_function_veri(h_root, g_labels_tensor)

	summ_gt_labels = tree_batch['summ_gt'].to('cpu')	
	summ_gt_labels_tensor = summ_gt_labels.clone().detach().to(device)
	if args.summ_loss_fn == 'nw':
		loss_function_summ = CrossEntropyLoss()
	else:
		class_weights = summ_weights.float()
		loss_function_summ = CrossEntropyLoss(weight=class_weights)
	loss_summ = loss_function_summ(summ_out, summ_gt_labels_tensor)
		
	loss = args.lambda1*loss_veri + args.lambda2*loss_summ
	if args.flood == 'y':
		loss = (loss-0.25).abs() + 0.25	
	
	if mode == "train":
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()
	
	return loss, pred_labels, pred_v, g_labels, pred_labels_summ, pred_v_summ, summ_gt_labels, att


def testing(test_model, tokenizer, dftest, test_trees, device):
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
	sit_tags = []

	test_model.eval()
	with torch.no_grad():
		for test in test_trees:						
			try:
				h_test_root, summ_out, att = test_model(
						test['f'].to(device),
						test['a'].to(device),
						# test['k'].to(device),
						test['node_order'].to(device),
						test['adjacency_list'].to(device),
						test['edge_order'].to(device),
						test['root_n'].to(device),
						test['sit_tag'].to(device)
				)
			except:
				continue
			lastlayer_attention = att[-1][0]
			lastlayer_attention = lastlayer_attention.to("cpu")
			a = torch.mean(lastlayer_attention, dim=0).squeeze(0)
			cls_attentions = a[0]
			token_attentions.append(cls_attentions)
			tokens = tokenizer.convert_ids_to_tokens(test['f'][0])
			tokenslist.append(tokens)
			num_tokens.append(int(torch.sum(test['a'][0]).item()))

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

			sit_tags.append(test['sit_tag'].item())						

			total += 1
	
	print(f'\nTotal Test trees evaluated: {total}')
	accuracy = accuracy_score(veri_ground, veri_predicted)
	print('Accuracy: %f' % accuracy)
	precision = precision_score(veri_ground, veri_predicted)
	print('Precision: %f' % precision)
	recall = recall_score(veri_ground, veri_predicted)
	print('Recall: %f' % recall)
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
					"summ_gt": summ_ground, 
					"sit_tag": sit_tags,
					"Tokens": tokenslist, 
					"Attentions": token_attentions, 
					"Numtokens": num_tokens}
				)
	dfsum[["Orig_Tweet", "Clean_Tweet", "Situational", "Summary_gt", "New_Summary_gt", "R1NR0"]] = dfsum.apply(lambda x : get_data(dftest, x), axis= 1)
	return dfsum

	
			  
def create_summary(args, dfsum, summary_filepath, pickle_path):
	MODEL_NAME = args.model_name
	TEST_FILE = args.test_file
	dfsum.sort_values(by=['summ_pred', 'summ_pred_prob'], inplace=True, ascending=False)
	dfsum = dfsum.reset_index(drop=True)

	groundtruth_summary_original = ""
	glist_original = dfsum[dfsum['Summary_gt']==1]['Orig_Tweet'].values
	groundtruth_summary_original = " .\n".join(glist_original)

	groundtruth_summary_cleaned  = ""		
	glist_cleaned = dfsum[dfsum['Summary_gt']==1]['Clean_Tweet'].values
	groundtruth_summary_cleaned = " .\n".join(glist_cleaned)
	
	dfsum.to_pickle(pickle_path)
	summ_orig, summ_clean, veri_prop, summary_length = generate_summary(250, dfsum)
	print(f'\nSummary generated for: {TEST_FILE}')

	content = ""
	content = content + "groundtruth_summary_original : \n\n" + groundtruth_summary_original + ".\n\n"
	content = content + "\ngroundtruth_summary_cleaned : \n\n" + groundtruth_summary_cleaned + ".\n\n"
	content = content + f'\nGenerated Summary Length = {summary_length}\n'
	content = content + "\nsummary_original_250 : \n\n" + summ_orig + ".\n\n"
	content = content + "\nsummary_cleaned_250 : \n\n" + summ_clean + ".\n\n"
					
		
	with open(summary_filepath, 'w') as f:
		f.write(content)

	print(f'\nVerified_Ratio of tweets in 250-words length summary for {TEST_FILE}: {veri_prop}\n\n')

	with open(MODEL_NAME + "_test_orig_250.txt",'w') as f:
		f.write(summ_orig)

	with open(MODEL_NAME + "_test_clean_250.txt",'w') as f:
		f.write(summ_clean)			

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
							print('\t' + prepare_results(metric, results_per_ref['p'][reference_id],
														results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
					print()
				else:
					print(prepare_results(metric, results['p'], results['r'], results['f']))
			print()

	

	print("\n\nFor Original Summary of length 250..")
	if summ_orig.strip() != "":
		os.chdir("twitie-tagger")
		return_code = subprocess.call("java -jar twitie_tag.jar models/gate-EN-twitter.model '../" + MODEL_NAME + "_test_orig_250.txt' > '../" + MODEL_NAME + "_output_orig_250.txt'", shell=True)
		os.chdir("..")

		num_words = 0
		content_words =0
		with open(MODEL_NAME + "_output_orig_250.txt", 'r') as f:
			for line in f:
				words = line.split()
				for word in words:
					l = word.split("_")
					# print(l)
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

	print("\n\nFor Cleaned Summary of length 250..")
	if summ_clean.strip() != "":
		os.chdir("twitie-tagger")
		return_code = subprocess.call("java -jar twitie_tag.jar models/gate-EN-twitter.model '../" + MODEL_NAME + "_test_clean_250.txt' > '../" + MODEL_NAME + "_output_clean_250.txt'", shell=True)  
		os.chdir("..")

		num_words = 0
		content_words =0
		with open(MODEL_NAME + "_output_clean_250.txt", 'r') as f:
			for line in f:
				words = line.split()
				for word in words:
					l = word.split("_")
					# print(l)
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
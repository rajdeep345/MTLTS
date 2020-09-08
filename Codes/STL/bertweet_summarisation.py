import os
import sys
import math
import argparse
import codecs
import random
import numpy as np
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

import re
import time
import datetime
import rouge
import textstat
import subprocess
import logging
logging.basicConfig(level=logging.ERROR)

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu_id', type=int, default=0)
	parser.add_argument('--model_name', type=str, default="BERTWEET")
	parser.add_argument('--in_features', type=int, default=768)
	parser.add_argument('--save_policy', type=str, default="loss")
	parser.add_argument('--loss_fn', type=str, default="w")
	parser.add_argument('--optim', type=str, default="adam")
	parser.add_argument('--l2', type=str, default="y")
	parser.add_argument('--wd', type=float, default=0.01)
	parser.add_argument('--use_dropout', type=str, default="n")
	parser.add_argument('--classifier_dropout', type=float, default=0.2)
	parser.add_argument('--iters', type=int, default=1)
	parser.add_argument('--bs', type=int, default=16)
	parser.add_argument('--seed', type=int, default=40)
	# parser.add_argument('--test_file', type=str, default="german")

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
	NUM_ITERATIONS = args.iters
	print(f'NUM_ITERATIONS = {NUM_ITERATIONS}')
	BATCH_SIZE = args.bs
	print(f'BATCH_SIZE = {BATCH_SIZE}')
	lr_list = [5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 5e-4]
	print(f'LEARNING_RATES = {str(lr_list)}')
	# TRAINABLE_LAYERS = [0,1,2,3,4,5,6,7,8,9,10,11]
	# print(f'TRAINABLE_LAYERS = {str(TRAINABLE_LAYERS)}')
	# TEST_FILE = args.test_file
	# print(f'\nTEST_FILE = {TEST_FILE}')

	# g_cpu = torch.Generator()
	# seed_val = g_cpu.seed()
	seed_val = args.seed
	print(f'\nSEED = {str(seed_val)}\n\n')

	# random.seed(seed_val)
	# numpy.random.seed(seed_val)
	# torch.manual_seed(seed_val)
	# torch.cuda.manual_seed(seed_val)

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

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', 199)  # or 199

d = {}
with open("/content/drive/My Drive/BTP_Chandana_Vishnu/summarization/CIKM/all_tweets_posteriors.txt") as file:
	for line in file:
		line = line.split("\t")
		d[int(line[1])] = eval(line[2])

#appending the situational feature to make is 41 dim vector
def get_features(x):
	features = []
	situ = float(x["Situational"])
	if IN_FEATURES == 769:		
		features.append(situ)
	elif IN_FEATURES == 808:
		features = d[x['Tweet_ID']]
	elif IN_FEATURES == 809:
		features = d[x['Tweet_ID']]
		features.append(situ)
	return features

# dfc = pd.read_pickle("/content/drive/My Drive/BTP_Chandana_Vishnu/summarization/Data/dfc_0625.pkl")
# dfg = pd.read_pickle("/content/drive/My Drive/BTP_Chandana_Vishnu/summarization/Data/dfg_08.pkl")
# dfo = pd.read_pickle("/content/drive/My Drive/BTP_Chandana_Vishnu/summarization/Data/dfo_069.pkl")
# dfs = pd.read_pickle("/content/drive/My Drive/BTP_Chandana_Vishnu/summarization/Data/dfs_0675.pkl")

# Manual SIT Labels
# dfc = pd.read_pickle("data/manual_sit/dfc_0625.pkl")
# dfg = pd.read_pickle("data/manual_sit/dfg_08.pkl")
# dfo = pd.read_pickle("data/manual_sit/dfo_069.pkl")
# dfs = pd.read_pickle("data/manual_sit/dfs_0675.pkl")

# Model SIT Labels
# dfc = pd.read_pickle("data/model_sit/dfc_065.pkl")
# dfg = pd.read_pickle("data/model_sit/dfg_08.pkl")
# dfo = pd.read_pickle("data/model_sit/dfo_069.pkl")
# dfs = pd.read_pickle("data/model_sit/dfs_0675.pkl")

# Model Bertweet
dfc = pd.read_pickle("/content/drive/My Drive/BTP_Chandana_Vishnu/Bertweetdata/dfc_065.pkl")
dfg = pd.read_pickle("/content/drive/My Drive/BTP_Chandana_Vishnu/Bertweetdata/dfg_08.pkl")
dfo = pd.read_pickle("/content/drive/My Drive/BTP_Chandana_Vishnu/Bertweetdata/dfo_069.pkl")
dfs = pd.read_pickle("/content/drive/My Drive/BTP_Chandana_Vishnu/Bertweetdata/dfs_0675.pkl")

dfc['features'] = dfc.apply(lambda x: get_features(x),axis =1)
dfg['features'] = dfg.apply(lambda x: get_features(x),axis =1)
dfo['features'] = dfo.apply(lambda x: get_features(x),axis =1)
dfs['features'] = dfs.apply(lambda x: get_features(x),axis =1)


def flat_accuracy(preds, labels):
	pred_flat = np.argmax(preds, axis=1).flatten()
	labels_flat = labels.flatten()
	return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
	'''
	Takes a time in seconds and returns a string hh:mm:ss
	'''
	# Round to the nearest second.
	elapsed_rounded = int(round((elapsed)))
	return str(datetime.timedelta(seconds=elapsed_rounded))


def softmax(z):
	assert len(z.shape) == 2
	s = np.max(z, axis=1)
	s = s[:, np.newaxis]
	e_x = np.exp(z - s)
	div = np.sum(e_x, axis=1)
	div = div[:, np.newaxis]
	return e_x / div


def removedup(seq):
	seen = set()
	seen_add = seen.add
	return [x for x in seq if not (x in seen or seen_add(x))]


def get_numwords(x):
	val = len(x['Clean_Tweet'].strip().split())
	if math.isnan(val):
		return 1
	else:
		return val


def generate_summary(numwords, dfsum):
	predicted_summary_orig = []
	predicted_summary_clean = []
	total_verified = 0
	summary_length = 0
	count = 0
	
	dfsum = dfsum[dfsum['Situational']==1].reset_index(drop=True)
	
	dfpred1 = dfsum[dfsum['pred']==1].reset_index(drop=True)
	if len(dfpred1) != 0:
		dfpred1.sort_values(by=['prob'], ascending=False, inplace=True)
		# list_1 = removedup(dfpred1['Clean_Tweet'].values)
		dfpred1.drop_duplicates(subset=['Clean_Tweet'], keep='first', inplace=True)
		dfpred1["Num_words"] = dfpred1.apply(lambda x : get_numwords(x), axis= 1)
		l1 = len(dfpred1)
		for i, row in dfpred1.iterrows():
			if summary_length < numwords:
				count += 1
				predicted_summary_orig.append(row['Orig_Tweet'])
				predicted_summary_clean.append(row['Clean_Tweet'])
				if int(row['R1NR0']) == 0:
					total_verified += 1
				summary_length += int(row['Num_words'])
			else:
				break

	if summary_length < numwords:
		dfpred0 = dfsum[dfsum['pred']==0].reset_index(drop=True)
		if len(dfpred0) != 0:
			dfpred0.sort_values(by=['prob'], ascending=True, inplace=True)
			# list_0 = removedup(dfpred0['Clean_Tweet'].values)
			dfpred0.drop_duplicates(subset=['Clean_Tweet'], keep='first', inplace=True)
			dfpred0["Num_words"] = dfpred0.apply(lambda x : get_numwords(x), axis= 1)
			l0 = len(dfpred0)
			for i, row in dfpred0.iterrows():
				if summary_length < numwords:
					count += 1
					predicted_summary_orig.append(row['Orig_Tweet'])
					predicted_summary_clean.append(row['Clean_Tweet'])
					if int(row['R1NR0']) == 0:
						total_verified += 1
					summary_length += int(row['Num_words'])
				else:
					break

	summ_orig = '.\n'.join(predicted_summary_orig).strip()
	summ_clean = '.\n'.join(predicted_summary_clean).strip()
	
	if count == 0:
		veri_prop = 0
	else:
		veri_prop = float(total_verified / count)
	
	return summ_orig, summ_clean, veri_prop, summary_length


def prepare_results(p, r, f):
	return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


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


def encode_plus(text,add_special_tokens = True, max_length = 32, pad_to_max_length = True,return_attention_mask = True):

	words = bpe.encode(text)

	if pad_to_max_length:
		if len(words.split())>(max_length-2):
			words = ' '.join(words.split()[:(max_length-2)])

	if add_special_tokens:
		subwords = '<s> ' + words + ' </s>'
	else:
		subwords = words

	input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
	tokens_len = len(input_ids)

	if pad_to_max_length:
		pad_len = max_length - tokens_len
		padding = [1]*pad_len
		input_ids.extend(padding)

	if return_attention_mask:
		attention_mask = [1]*tokens_len
		attention_mask.extend([0]*pad_len)

	return { 'input_ids': torch.tensor([input_ids]),
	'attention_mask': torch.tensor([attention_mask]) }

#load models
tweetconfig = RobertaConfig.from_pretrained(
	"/content/BERTweet_base_transformers/config.json",

)

# Load the tokenizer.
# print(f'\nLoading the {MODEL_NAME} tokenizer..')
# if MODEL_NAME == 'BERT':	
# 	tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# elif MODEL_NAME == 'ROBERTA':
# 	tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Load BPE encoder 
parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes', 
	default="BERTweet_base_transformers/bpe.codes",
	required=False,
	type=str,  
	help='path to fastBPE BPE'
)
args = parser.parse_args()
bpe = fastBPE(args)

vocab = Dictionary()
vocab.add_from_file("/content/BERTweet_base_transformers/dict.txt")

class BertForSequenceClassificationModified(BertPreTrainedModel):
	if MODEL_NAME == 'ROBERTA':
		config_class = RobertaConfig
		base_model_prefix = "roberta"

	def __init__(self, config):
		super().__init__(config)		
		self.num_labels = config.num_labels
		
		if MODEL_NAME == 'BERT':
			self.bert = BertModel(config)		
			# self.dropout = nn.Dropout(config.hidden_dropout_prob)
			# self.dropout1 = nn.Dropout(0.2)		
			self.fc = nn.Linear(IN_FEATURES, OUT_FEATURES)
			self.classifier = nn.Linear(OUT_FEATURES, config.num_labels)
		elif MODEL_NAME == 'ROBERTA':
			# self.roberta = RobertaModel(config)
			self.bert = RobertaModel(config)
			# self.classifier = RobertaClassificationHead(config)
			self.fc = nn.Linear(IN_FEATURES, OUT_FEATURES)
			self.classifier = nn.Linear(OUT_FEATURES, config.num_labels)
		elif MODEL_NAME == 'BERTWEET':
			self.bert = RobertaModel.from_pretrained("/content/BERTweet_base_transformers/model.bin",config= tweetconfig)
			self.fc = nn.Linear(IN_FEATURES, OUT_FEATURES)
			self.classifier = nn.Linear(OUT_FEATURES, tweetconfig.num_labels)
		
		# torch.nn.init.xavier_normal(self.fc.weight)
		# torch.nn.init.xavier_normal(self.classifier.weight)
		
		self.init_weights()
	
	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		labels=None,
		features= None,
		classweights = None,
	):
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)

		sequence_output = outputs[0]
		cls = sequence_output[:, 0]
		
		if IN_FEATURES == 768:
			output_vectors = cls
		else:
			output_vectors = torch.cat([cls, features], axis=1)
		
		logits1 = self.fc(output_vectors)
		
		# logits1 = self.dropout1(logits1)
		
		logits = self.classifier(logits1)

		outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

		if labels is not None:
			if self.num_labels == 1:
				#  We are doing regression
				loss_fct = MSELoss()
				loss = loss_fct(logits.view(-1), labels.view(-1))
			else:
				class_weights = torch.FloatTensor(classweights).cuda(gpu_id)
				if LOSS_FN == 'w':
					loss_fct = CrossEntropyLoss(weight=class_weights)
				else:
					loss_fct = CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			outputs = (loss,) + outputs

		return outputs  # (loss), logits, (hidden_states), (attentions)


dfts = pd.concat([dfc, dfg, dfo], axis =0).reset_index(drop=True)
dftc = pd.concat([dfs, dfg, dfo], axis =0).reset_index(drop=True)
dftg = pd.concat([dfc, dfs, dfo], axis =0).reset_index(drop=True)
dfto = pd.concat([dfc, dfg, dfs], axis =0).reset_index(drop=True)

place_summary = {
	"sydneysiege": [dfts, dfs], 
	"charliehebdo": [dftc, dfc], 
	"germanwings": [dftg, dfg], 
	"ottawashooting":[dfto, dfo]
	}

for lr in lr_list:
	print("\n\n\nTraining with LR: ", lr)
	for place, dflist in place_summary.items():

		print(f'\nTraining for {place}')

		random.seed(seed_val)
		np.random.seed(seed_val)
		torch.manual_seed(seed_val)
		torch.cuda.manual_seed(seed_val)

		dft = dflist[0]
		dft = dft.sample(frac=1).reset_index(drop=True)
		
		path = "./Models/"
		name = path + "stl_summarisation_" + place + "_" + str(IN_FEATURES) + "_feat_" + MODEL_NAME + ".pt"

		#varried wrt to combination
		y =list(dft['New_Summary_gt'].values)
		y = np.array(y)
		class_weights = compute_class_weight('balanced', np.unique(y),y)
		
		sentences = dft.New_Cleaned_Tweet
		labels = dft.New_Summary_gt
		features = dft.features
		input_ids = []
		attention_masks = []
		
		for sent in sentences:
			encoded_dict = encode_plus(
								sent,							# Sentence to encode.
								add_special_tokens = True,		# Add '[CLS]' and '[SEP]'
								max_length = 32,				# Pad & truncate all sentences.
								pad_to_max_length = True,		# pad to max length
								return_attention_mask = True,	# Construct attn. masks.
						)
			
			# Add the encoded sentence to the list.    
			input_ids.append(encoded_dict['input_ids'])
			
			# And its attention mask (simply differentiates padding from non-padding).
			attention_masks.append(encoded_dict['attention_mask'])

		# Convert the lists into tensors.
		input_ids = torch.cat(input_ids, dim=0)
		attention_masks = torch.cat(attention_masks, dim=0)
		labels = torch.tensor(labels)
		features  = torch.tensor(features)
		
		dataset = TensorDataset(input_ids, attention_masks, labels, features)

		train_size = int(0.9 * len(dataset))
		val_size = len(dataset) - train_size

		# Divide the dataset by randomly selecting samples.
		train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

		print('{:>5,} training samples'.format(train_size))
		print('{:>5,} validation samples'.format(val_size))

		# Create the DataLoaders for our training and validation sets.
		# We'll take training samples in random order. 
		train_dataloader = DataLoader(
					train_dataset,							# The training samples.
					sampler = RandomSampler(train_dataset),	# Select batches randomly
					batch_size = BATCH_SIZE					# Trains with this batch size.
				)

		# For validation the order doesn't matter, so we'll just read them sequentially.
		validation_dataloader = DataLoader(
					val_dataset,								# The validation samples.
					sampler = SequentialSampler(val_dataset),	# Pull out batches sequentially.
					batch_size = BATCH_SIZE						# Evaluate with this batch size.
				)

		if MODEL_NAME == 'BERT':			
			model = BertForSequenceClassificationModified.from_pretrained(
			"bert-base-cased",				# Use the 12-layer BERT model, with an cased vocab.
			num_labels = 2,					# The number of output labels --2 for binary classification.
											# You can increase this for multi-class tasks.   
			output_attentions = False,		# Whether the model returns attentions weights.
			output_hidden_states = False,	# Whether the model returns all hidden-states.
			)
		elif MODEL_NAME == 'ROBERTA':
			model = BertForSequenceClassificationModified.from_pretrained(
			"roberta-base",					# Use the 12-layer RoBERTa model, with base vocab.
			num_labels = 2,					# The number of output labels --2 for binary classification.
											# You can increase this for multi-class tasks.   
			output_attentions = False,		# Whether the model returns attentions weights.
			output_hidden_states = False,	# Whether the model returns all hidden-states.
			)
		elif MODEL_NAME == 'BERTWEET':
			tweetconfig.output_attentions = False
			tweetconfig.output_hidden_states = False
			model = BertForSequenceClassificationModified(tweetconfig)
		# Tell pytorch to run this model on the GPU.
		model.cuda(gpu_id)
		if OPTIM == 'adam':
			if L2_REGULARIZER == 'n':
				optimizer = torch.optim.Adam(model.parameters(), lr=lr)
			else:
				print(f"L2_REGULARIZER = y and WEIGHT_DECAY = {WEIGHT_DECAY}")
				optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
		else:
			optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)

		# optimizer = torch.optim.Adam(model.parameters(),
		# 			lr = lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
		# 			eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
		# 			)

		epochs = NUM_ITERATIONS

		# Total number of training steps is [number of batches] x [number of epochs]. 
		# (Note that this is not the same as the number of training samples).
		# total_steps = len(train_dataloader) * epochs

		# Create the learning rate scheduler.
		# scheduler = get_linear_schedule_with_warmup(optimizer, 
		# 											num_warmup_steps = 0, # Default value in run_glue.py
		# 											num_training_steps = total_steps)


		training_stats = []

		# Measure the total training time for the whole run.
		total_t0 = time.time()

		# For each epoch...
		prev_loss = math.inf
		prev_acc = 0
		for epoch_i in range(0, epochs):
			
			# ========================================
			#               Training
			# ========================================
			
			# Perform one full pass over the training set.

			print("")
			print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
			print('Training...')

			# Measure how long the training epoch takes.
			t0 = time.time()

			# Reset the total loss for this epoch.
			total_train_loss = 0
			model.train()

			# For each batch of training data...
			for step, batch in enumerate(train_dataloader):

				# Progress update every 40 batches.
				if step % 40 == 0 and not step == 0:
					# Calculate elapsed time in minutes.
					elapsed = format_time(time.time() - t0)
					
					# Report progress.
					print('  Batch {:>5,}  of  {:>5,}.  Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

				b_input_ids = batch[0].to(device)
				b_input_mask = batch[1].to(device)
				b_labels = batch[2].to(device)
				b_features = batch[3].to(device)
				
				model.zero_grad()        
				loss, logits = model(b_input_ids, 
									token_type_ids=None, 
									attention_mask=b_input_mask, 
									labels=b_labels,
									features = b_features,
									classweights = class_weights)

				total_train_loss += loss.item()

				# Perform a backward pass to calculate the gradients.
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				optimizer.step()

				# Update the learning rate.
				# scheduler.step()

			# Calculate the average loss over all of the batches.
			avg_train_loss = total_train_loss / len(train_dataloader)            
			
			# Measure how long this epoch took.
			training_time = format_time(time.time() - t0)

			print("")
			print("Average training loss: {0:.2f}".format(avg_train_loss))
			print("Training epcoh took: {:}".format(training_time))
				
			# ========================================
			#               Validation
			# ========================================
			# After the completion of each training epoch, measure our performance on
			# our validation set.

			print("")
			print("Running Validation...")

			t0 = time.time()

			# Put the model in evaluation mode--the dropout layers behave differently
			# during evaluation.
			model.eval()

			# Tracking variables 
			total_eval_accuracy = 0
			total_eval_loss = 0
			nb_eval_steps = 0

			# Evaluate data for one epoch
			for batch in validation_dataloader:				
				b_input_ids = batch[0].to(device)
				b_input_mask = batch[1].to(device)
				b_labels = batch[2].to(device)
				b_features = batch[3].to(device)
				
				# Tell pytorch not to bother with constructing the compute graph during
				# the forward pass, since this is only needed for backprop (training).
				with torch.no_grad():
					(loss, logits) = model(b_input_ids, 
										token_type_ids=None, 
										attention_mask=b_input_mask,
										labels=b_labels,
										features = b_features,
										classweights = class_weights)
					
				# Accumulate the validation loss.
				total_eval_loss += loss.item()

				# Move logits and labels to CPU
				logits = logits.detach().cpu().numpy()
				label_ids = b_labels.to('cpu').numpy()

				# Calculate the accuracy for this batch of test sentences, and
				# accumulate it over all batches.
				total_eval_accuracy += flat_accuracy(logits, label_ids)
				

			# Report the final accuracy for this validation run.
			avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
			print("Validation Accuracy: {0:.2f}".format(avg_val_accuracy))

			# Calculate the average loss over all of the batches.
			avg_val_loss = total_eval_loss / len(validation_dataloader)
			
			# Measure how long the validation run took.
			validation_time = format_time(time.time() - t0)
			
			print("Validation Loss: {0:.2f}".format(avg_val_loss))
			print("Validation took: {:}".format(validation_time))

			if MODEL_SAVING_POLICY == "acc":
				if (prev_acc <= avg_val_accuracy):
					save_model(model, name, avg_val_accuracy, avg_val_loss)
					prev_acc = avg_val_accuracy
			else:			
				if (prev_loss >= avg_val_loss):
					save_model(model, name, avg_val_accuracy, avg_val_loss)
					prev_loss = avg_val_loss
			
			# Record all statistics from this epoch.
			training_stats.append(
				{
					'epoch': epoch_i + 1,
					'Training Loss': avg_train_loss,
					'Valid. Loss': avg_val_loss,
					'Valid. Accur.': avg_val_accuracy,
					'Training Time': training_time,
					'Validation Time': validation_time
				}
			)

		print("\nTraining complete!")

		print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

		if MODEL_NAME == 'BERT':			
			test_model = BertForSequenceClassificationModified.from_pretrained(
			"bert-base-cased",				# Use the 12-layer BERT model, with an cased vocab.
			num_labels = 2,					# The number of output labels --2 for binary classification.
											# You can increase this for multi-class tasks.   
			output_attentions = True,		# Whether the model returns attentions weights.
			output_hidden_states = True,	# Whether the model returns all hidden-states.
			)
		elif MODEL_NAME == 'ROBERTA':
			test_model = BertForSequenceClassificationModified.from_pretrained(
			"roberta-base",					# Use the 12-layer RoBERTa model, with base vocab.
			num_labels = 2,					# The number of output labels --2 for binary classification.
											# You can increase this for multi-class tasks.   
			output_attentions = True,		# Whether the model returns attentions weights.
			output_hidden_states = True,	# Whether the model returns all hidden-states.
			)
		elif MODEL_NAME == 'BERTWEET':
			tweetconfig.output_attentions = True
			tweetconfig.output_hidden_states = True
			test_model = BertForSequenceClassificationModified(tweetconfig)

		test_model.cuda(gpu_id)

		#testing
		dftest = dflist[1]
		# Report the number of sentences.
		print('Number of test sentences: {:,}\n'.format(dftest.shape[0]))

		# Create sentence and label lists
		sentences = dftest.New_Cleaned_Tweet.values
		labels = dftest.New_Summary_gt.values
		features = dftest.features
		# Tokenize all of the sentences and map the tokens to thier word IDs.
		input_ids = []
		attention_masks = []
		tokenslist = []
		num_tokens = []

		# For every sentence...
		for sent in sentences:
			subwords = '<s> ' + bpe.encode(sent) + ' </s>'
			tokenslist.append(subwords.split())
			encoded_dict = encode_plus(
								sent,							# Sentence to encode.
								add_special_tokens = True,		# Add '[CLS]' and '[SEP]'
								max_length = 32,				# Pad & truncate all sentences.
								pad_to_max_length = True,		# pad to max length
								return_attention_mask = True,	# Construct attn. masks.
						)
			
			input_ids.append(encoded_dict['input_ids'])
			
			# And its attention mask (simply differentiates padding from non-padding).
			attention_masks.append(encoded_dict['attention_mask'])
			num_tokens.append(torch.sum(encoded_dict["attention_mask"], dim=1).item())

		# Convert the lists into tensors.
		input_ids = torch.cat(input_ids, dim=0)
		attention_masks = torch.cat(attention_masks, dim=0)
		labels = torch.tensor(labels)
		features = torch.tensor(features)
		
		# Set the batch size.
		batch_size = 32

		# Create the DataLoader.
		prediction_data = TensorDataset(input_ids, attention_masks, labels, features)
		prediction_sampler = SequentialSampler(prediction_data)
		prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

		print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))
		
		load_model(test_model, name)
		
		# Put model in evaluation mode
		test_model.eval()

		# Tracking variables 
		predictions , true_labels = [], []
		token_attentions =[]

		# Predict 
		for batch in prediction_dataloader:
			# Add batch to GPU
			batch = tuple(t.to(device) for t in batch)
			
			# Unpack the inputs from our dataloader
			b_input_ids, b_input_mask, b_labels, features = batch
			
			# Telling the model not to compute or store gradients, saving memory and speeding up prediction
			with torch.no_grad():
				# Forward pass, calculate logit predictions
				outputs = test_model(b_input_ids, 
									token_type_ids=None, 
									attention_mask=b_input_mask, 
									features = features
								)

			logits = outputs[0]

			lastlayer_attention = outputs[-1][-1]
			lastlayer_attention = lastlayer_attention.to("cpu")
			a = torch.mean(lastlayer_attention, dim=1)
			cls_attentions = a[:, 0, :]
			token_attentions.append(cls_attentions)

			# Move logits and labels to CPU
			logits = logits.detach().cpu().numpy()
			label_ids = b_labels.to('cpu').numpy()
			
			# Store predictions and true labels
			predictions.append(logits)
			true_labels.append(label_ids)

		print('Training DONE...')

		flat_predictions = np.concatenate(predictions, axis=0)
		f_predictions = np.concatenate(predictions, axis=0)
		clsattentions = np.concatenate(token_attentions, axis =0)

		# For each sample, pick the label (0 or 1) with the higher score.
		flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

		# Combine the correct labels for each batch into a single list.
		flat_true_labels = np.concatenate(true_labels, axis=0)

		print("\n\n Testing results for place: ", place)
		print(classification_report(flat_true_labels, flat_predictions, digits=6))

		#summary generation
		dfp = dftest.copy()
		dfsum = pd.DataFrame({'Tweet_ID': dfp.Tweet_ID,
							'Orig_Tweet': dfp.Orig_Tweet,
							'Clean_Tweet': dfp.Clean_Tweet,
							'Summary_gt': dfp["Summary_gt"],
							'New_Summary_gt': dfp.New_Summary_gt,
							'New_Cleaned_Tweet':dfp.New_Cleaned_Tweet,
							'pred': flat_predictions, 
							'Situational': dfp.Situational,
							'R1NR0': dfp.R1NR0,
							"Tokens": tokenslist,
							"Attentions": list(clsattentions),
							"Numtokens": num_tokens})
		
		prob = softmax(f_predictions)
		# print(prob)
		probabilities = []
		for i in range(len(dfsum)):
			probabilities.append(prob[i][dfsum.iloc[i]['pred']])
			
		dfsum['prob'] = probabilities
		dfsum.sort_values(by=['pred','prob'], inplace=True, ascending=False)
		dfsum = dfsum.reset_index(drop=True)
		# print(dfsum)
		# print(dfsum.to_csv(sep='\t', index=False))
		
		groundtruth_summary_original = ""
		glist_original = dfp[dfp['Summary_gt']==1]['Orig_Tweet'].values
		groundtruth_summary_original = " .\n".join(glist_original)

		groundtruth_summary_cleaned  = ""		
		glist_cleaned = dfp[dfp['Summary_gt']==1]['Clean_Tweet'].values
		groundtruth_summary_cleaned = " .\n".join(glist_cleaned)
		
		if L2_REGULARIZER == 'y':
			filepath = "STL_" + MODEL_NAME + "_" + str(IN_FEATURES) + "_" + str(lr) + "_L2y_" + str(WEIGHT_DECAY) + '_' + place + ".pkl"
		else:
			filepath = "STL_" + MODEL_NAME + "_" + str(IN_FEATURES) + "_" + str(lr) + "_L2n_" + place + ".pkl"
		dfsum.to_pickle(filepath)
		
		summ_orig, summ_clean, veri_prop, summary_length = generate_summary(250, dfsum)
		print(f'\nSummary generated for: {place}')

		content = ""
		content = content + "groundtruth_summary_original : \n\n" + groundtruth_summary_original + ".\n\n"
		content = content + "\ngroundtruth_summary_cleaned : \n\n" + groundtruth_summary_cleaned + ".\n\n"
		content = content + f'\nGenerated Summary Length = {summary_length}\n'
		content = content + "\nsummary_original_250 : \n\n" + summ_orig + ".\n\n"
		content = content + "\nsummary_cleaned_250 : \n\n" + summ_clean + ".\n\n"
		# print(content)
		
		if L2_REGULARIZER == 'y':
			summary_filepath = "STL_" + MODEL_NAME + "_" + str(IN_FEATURES) + "_" + str(lr) + "_L2y_" + str(WEIGHT_DECAY) + '_' + place + "_summary.txt"
		else:
			summary_filepath = "STL_" + MODEL_NAME + "_" + str(IN_FEATURES) + "_" + str(lr) + "_L2n_" + place + "_summary.txt"
		
		with open(summary_filepath, 'w') as f:
			f.write(content)

		print(f'\nVerified_Ratio of tweets in 250-words length summary for {place}: {veri_prop}\n\n')

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
								print('\t' + prepare_results(results_per_ref['p'][reference_id],
															results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
						print()
					else:
						print(prepare_results(results['p'], results['r'], results['f']))
				print()

		

		print("\n\nFor Original Summary of length 250..")
		if summ_orig.strip() != "":
			os.chdir("twitie-tagger")
			return_code = subprocess.call("/usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java -jar twitie_tag.jar models/gate-EN-twitter.model '../" + MODEL_NAME + "_test_orig_250.txt' > '../" + MODEL_NAME + "_output_orig_250.txt'", shell=True)
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
			return_code = subprocess.call("/usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java -jar twitie_tag.jar models/gate-EN-twitter.model '../" + MODEL_NAME + "_test_clean_250.txt' > '../" + MODEL_NAME + "_output_clean_250.txt'", shell=True)  
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

		print(f'\n\nDone for place: {place}\n\n\n')



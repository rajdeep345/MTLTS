# SummaRuNNer with BERT main.py

#!/usr/bin/env python3

import subprocess as sp
import os
import json
import models
import utils
import argparse,random,logging,numpy,os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from torch.nn.utils import clip_grad_norm
from time import time
from tqdm import tqdm
from collections import namedtuple
import pandas as pd
from pprint import pprint


dfc = pd.read_pickle("../data/features/summary_dataframes/dfc_0.57.pkl")
dfg = pd.read_pickle("../data/features/summary_dataframes/dfg_0.72.pkl")
dfo = pd.read_pickle("../data/features/summary_dataframes/dfo_0.6.pkl")
dfs = pd.read_pickle("../data/features/summary_dataframes/dfs_0.6.pkl")
# dfmain = pd.concat([dfc,dfg,dfo,dfs],ignore_index= True)
# dfmain['Tweet_ID'] = dfmain['Tweet_ID'].astype(int)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
parser = argparse.ArgumentParser(description='extractive summary')
# model
parser.add_argument('-save_dir',type=str,default='checkpoints/')
parser.add_argument('-embed_dim',type=int,default=100)
parser.add_argument('-embed_num',type=int,default=100)
parser.add_argument('-pos_dim',type=int,default=100)
parser.add_argument('-pos_num',type=int,default=300)
parser.add_argument('-seg_num',type=int,default=10)
parser.add_argument('-kernel_num',type=int,default=100)
parser.add_argument('-kernel_sizes',type=str,default='3,4,5')
parser.add_argument('-model',type=str,default='RNN_RNN')
parser.add_argument('-hidden_size',type=int,default=200)
# train
parser.add_argument('-lr',type=float,default=1e-5)
parser.add_argument('-batch_size',type=int,default=32)
parser.add_argument('-epochs',type=int,default=15)
parser.add_argument('-seed',type=int,default=1)
parser.add_argument('-train_dir',type=str,default='data/train.json')
parser.add_argument('-val_dir',type=str,default='data/val.json')
parser.add_argument('-embedding',type=str,default='data/embedding.npz')
parser.add_argument('-word2id',type=str,default='data/word2id.json')
parser.add_argument('-report_every',type=int,default=2)
parser.add_argument('-seq_trunc',type=int,default=50)
parser.add_argument('-max_norm',type=float,default=1.0)
# test
parser.add_argument('-load_dir',type=str,default='checkpoints/RNN_RNN_seed_1.pt')
parser.add_argument('-test_dir',type=str,default='data/test.json')
parser.add_argument('-ref',type=str,default='outputs/ref')
parser.add_argument('-hyp',type=str,default='outputs/hyp')
parser.add_argument('-filename',type=str,default='x.txt') # TextFile to be summarized
parser.add_argument('-topk',type=int,default=20)
# device
parser.add_argument('-device',type=int)
# option
parser.add_argument('-test',action='store_true')
parser.add_argument('-debug',action='store_true')
parser.add_argument('-predict',action='store_true')
args = parser.parse_args()
use_gpu = args.device is not None

if torch.cuda.is_available() and not use_gpu:
	print("WARNING: You have a CUDA device, should run with -device 0")

# set cuda device and seed
if use_gpu:
	torch.cuda.set_device(args.device)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed)


def eval(net,vocab,data_iter,criterion):
	with torch.no_grad():
		net.eval()
		total_loss = 0
		batch_num = 0
		for batch in data_iter:
			input_ids,attention_masks,targets,_,doc_lens = vocab.make_features(batch)
			input_ids,attention_masks,targets = Variable(input_ids),Variable(attention_masks), Variable(targets.float())
			if use_gpu:
				targets = targets.cuda()
				input_ids = input_ids.cuda()
				attention_masks = attention_masks.cuda()
			probs = net(input_ids,attention_masks,doc_lens)
			loss = criterion(probs,targets)
			total_loss += loss.item()
			batch_num += 1
		loss = total_loss / batch_num
		del targets
		del input_ids
		del attention_masks
		torch.cuda.empty_cache()
		net.train()
	return loss

def train():
	logging.info('Loading vocab,train and val dataset.Wait a second,please')
	pp = 3
	vocab = utils.Vocab()

	with open(args.train_dir) as f:
		examples = json.load(f)
	# print(len(examples))
	train_dataset = utils.Dataset(examples)
	# print(train_dataset)
	with open(args.val_dir) as f:
		examples = json.load(f)
	val_dataset = utils.Dataset(examples)

	# update args
	args.kernel_sizes = [int(ks) for ks in args.kernel_sizes.split(',')]
	acc_steps = 16
	# build model
	net = getattr(models,args.model)(args)
	if use_gpu:
		net.cuda()
	# load dataset
	train_iter = DataLoader(dataset=train_dataset,
			batch_size=args.batch_size,
			shuffle=True)
	val_iter = DataLoader(dataset=val_dataset,
			batch_size=args.batch_size,
			shuffle=False)
	# loss function
	criterion = nn.BCELoss()
	# model info
	print(net)
	params = sum(p.numel() for p in list(net.parameters())) / 1e6
	print('#Params: %.1fM' % (params))

	min_loss = float('inf')
	optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
	net.train()

	t1 = time()
	checkpp = 0
	for epoch in range(1,args.epochs+1):
		if(checkpp==pp):
			break
		optimizer.zero_grad()
		t_loss = 0
		s_loss = 0
		from pprint import pprint
		for i,batch in enumerate(train_iter):
			# print(batch)
			input_ids,attention_masks,targets,_,doc_lens = vocab.make_features(batch)
			input_ids,attention_masks,targets = Variable(input_ids),Variable(attention_masks), Variable(targets.float())
			if use_gpu:
				input_ids = input_ids.cuda()
				attention_masks = attention_masks.cuda()
			   
			probs = net(input_ids,attention_masks,doc_lens)
			if use_gpu:
				targets = targets.cuda()
		   
			loss = criterion(probs,targets)
			t_loss = t_loss+loss.item()
			loss = loss / acc_steps
			s_loss = s_loss+1
			loss.backward()
			clip_grad_norm(net.parameters(), args.max_norm)
			if(((i+1) % acc_steps == 0) or (i==800)):
				optimizer.step()
				optimizer.zero_grad()
			if args.debug:
				print('Batch ID:%d Loss:%f' %(i,loss.data[0]))
				continue
			if( (i % args.report_every == 0) and (i!=0)):
				cur_loss = eval(net,vocab,val_iter,criterion)
				train_loss = t_loss/s_loss
				t_loss = 0
				s_loss = 0
				if cur_loss < min_loss:
					checkpp = 0
					min_loss = cur_loss
					best_path = net.save()
				else:
					checkpp = checkpp+1

				logging.info('Epoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f training loss: %f'
						% (epoch,min_loss,cur_loss,train_loss))

	t2 = time()
	logging.info('Total Cost:%f h'%((t2-t1)/3600))

'''
def Get_tweetID(sentence):
	# same tweet content different IDs
	x = dfmain[dfmain['Clean_Tweet']==sentence].Tweet_ID.iloc[0]
	return int(x)
'''

def test():
	aggregate_dict = {int(tweetd):{'sum':0,'count':0} for tweetd in dfs.Tweet_ID}
	vocab = utils.Vocab()

	with open(args.test_dir) as f:
		# examples = [json.loads(line) for line in f]
		examples = json.load(f)
	test_dataset = utils.Dataset(examples)

	test_iter = DataLoader(dataset=test_dataset,
							batch_size=args.batch_size,
							shuffle=False)
	if use_gpu:
		checkpoint = torch.load(args.load_dir)
	else:
		checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

	# checkpoint['args']['device'] saves the device used as train time
	# if at test time, we are using a CPU, we must override device to None
	if not use_gpu:
		checkpoint['args'].device = None
	net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
	net.load_state_dict(checkpoint['model'])
	if use_gpu:
		net.cuda()
	net.eval()

	doc_num = len(test_dataset)
	time_cost = 0
	file_id = 1
	for batch in tqdm(test_iter):
		input_ids,attention_masks,targets,summaries,doc_lens  = vocab.make_features(batch)
		input_ids,attention_masks,targets = Variable(input_ids),Variable(attention_masks), Variable(targets.float())
		t1 = time()
		if use_gpu:
			input_ids = input_ids.cuda()
			attention_masks = attention_masks.cuda()

			probs = net(input_ids,attention_masks,doc_lens)
		else:
			probs = net(input_ids,attention_masks,doc_lens)
		t2 = time()
		time_cost += t2 - t1
		start = 0
		for doc_id,doc_len in enumerate(doc_lens):
			# print(doc_len)
			stop = start + doc_len
			prob = probs[start:stop]
			topk = min(args.topk,doc_len)
			topk = doc_len
			topk_indices = prob.topk(topk)[1].cpu().data.numpy()
			# topk_indices.sort()
			doc = batch['doc'][doc_id].split('\n')[:doc_len]
			tweeidlist = batch['tweetid'][doc_id].split('\n')[:doc_len]
			for index in topk_indices:
				sentence = doc[index]
				# tid = Get_tweetID(sentence)
				tid = int(tweeidlist[index])
				aggregate_dict[tid]['sum']+=prob[index].item()
				aggregate_dict[tid]['count']+=1
			hyp = [str(prob[index].item()) + "\t"+ doc[index] for index in topk_indices]
			ref = summaries[doc_id]
			# with open(os.path.join(args.ref,str(file_id)+'.txt'), 'w') as f:
			#     f.write(ref)
			with open(os.path.join(args.hyp,str(file_id)+'.txt'), 'w') as f:
				f.write('\n'.join(hyp))
			start = stop
			file_id = file_id + 1
		del input_ids
		del attention_masks
		torch.cuda.empty_cache()

	pprint(aggregate_dict)
	print('Speed: %.2f docs / s' % (doc_num / time_cost))


def predict(examples):

	vocab = utils.Vocab()
	pred_dataset = utils.Dataset(examples)

	pred_iter = DataLoader(dataset=pred_dataset,
							batch_size=args.batch_size,
							shuffle=False)
	if use_gpu:
		checkpoint = torch.load(args.load_dir)
	else:
		checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

	# checkpoint['args']['device'] saves the device used as train time
	# if at test time, we are using a CPU, we must override device to None
	if not use_gpu:
		checkpoint['args'].device = None
	net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
	net.load_state_dict(checkpoint['model'])
	if use_gpu:
		net.cuda()
	net.eval()

	doc_num = len(pred_dataset)
	time_cost = 0
	file_id = 1
	for batch in tqdm(pred_iter):
		input_ids,attention_masks, doc_lens = vocab.make_predict_features(batch)
		t1 = time()
		if use_gpu:
			probs = net(input_ids,attention_masks,doc_lens)
		else:
			input_ids,attention_masks = Variable(input_ids),Variable(attention_masks)
			probs = net(input_ids,attention_masks, doc_lens)
		t2 = time()
		time_cost += t2 - t1
		start = 0
		for doc_id,doc_len in enumerate(doc_lens):
			stop = start + doc_len
			prob = probs[start:stop]
			topk = min(args.topk,doc_len)
			topk_indices = prob.topk(topk)[1].cpu().data.numpy()
			topk_indices.sort()
			
			doc = batch[doc_id].split('. ')[:doc_len]
			hyp = [doc[index] for index in topk_indices]
			with open(os.path.join(args.hyp,str(file_id)+'.txt'), 'w') as f:
				f.write('. '.join(hyp))
			start = stop
			file_id = file_id + 1
	print('Speed: %.2f docs / s' % (doc_num / time_cost))

if __name__=='__main__':
	if args.test:
		test()
	elif args.predict:
		with open(args.filename) as file:
			bod = [file.read()]
		predict(bod)
	else:
		train()

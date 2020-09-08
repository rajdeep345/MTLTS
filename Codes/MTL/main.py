import argparse
import random
import torch
import math
import numpy as np
from Model import *
from data import *
from TreeData import *
from Utils import *
from torch.utils.data import Dataset, IterableDataset, DataLoader, TensorDataset
from torch.utils.data import random_split, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# global gpu_id, MODEL_NAME, IN_FEATURES, USE_SITU, OUT_FEATURES,	MODEL_SAVING_POLICY, VERI_LOSS_FN, SUMM_LOSS_FN, LAMBDA1, LAMBDA2, FLOOD, OPTIM, L2_REGULARIZER, WEIGHT_DECAY,	USE_DROPOUT,	CLASSIFIER_DROPOUT,	TREE_VERSION,NUM_ITERATIONS, BATCH_SIZE, lr_list, TRAINABLE_LAYERS, TEST_FILE, seed_val 



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu_id', type=int, default=0)
	parser.add_argument('--model_name', type=str, default="BERT") 
	parser.add_argument('--in_features', type=int, default=768)
	parser.add_argument('--use_situ_tag', type=str, default="n")
	parser.add_argument('--save_policy', type=str, default="loss")
	parser.add_argument('--veri_loss_fn', type=str, default="nw")
	parser.add_argument('--summ_loss_fn', type=str, default="w")
	parser.add_argument('--optim', type=str, default="adam")
	parser.add_argument('--l2', type=str, default="y")
	parser.add_argument('--wd', type=float, default=0.01)
	parser.add_argument('--use_dropout', type=str, default="n")
	parser.add_argument('--classifier_dropout', type=float, default=0.2)
	parser.add_argument('--tree', type=str, default="new")
	parser.add_argument('--iters', type=int, default=1)
	parser.add_argument('--bs', type=int, default=16)
	parser.add_argument('--seed', type=int, default=40)
	parser.add_argument('--lambda1', type=float, default=0.5)
	parser.add_argument('--lambda2', type=float, default=0.5)
	parser.add_argument('--flood', type=str, default="n")
	parser.add_argument('--test_file', type=str, default="german")

	args = parser.parse_args()

	gpu_id = args.gpu_id
	print(f'GPU_ID = {gpu_id}\n')   
	MODEL_NAME = args.model_name
	print(f'MODEL_NAME = {MODEL_NAME}')
	IN_FEATURES = args.in_features
	print(f'IN_FEATURES = {IN_FEATURES}')
	USE_SITU = args.use_situ_tag
	print(f'USE_SITU = {USE_SITU}')
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
	TREE_VERSION = args.tree
	print(f'TREE_VERSION = {TREE_VERSION}')
	NUM_ITERATIONS = args.iters
	print(f'NUM_ITERATIONS = {NUM_ITERATIONS}')
	BATCH_SIZE = args.bs
	print(f'BATCH_SIZE = {BATCH_SIZE}')
	# lr_list = [5e-6, 1e-5, 2e-5, 5e-5, 1e-4]
	lr_list = [2e-5]
	print(f'LEARNING_RATES = {str(lr_list)}')
	TRAINABLE_LAYERS = [0,1,2,3,4,5,6,7,8,9,10,11]
	print(f'TRAINABLE_LAYERS = {str(TRAINABLE_LAYERS)}')
	TEST_FILE = args.test_file
	print(f'\nTEST_FILE = {TEST_FILE}')
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
	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed(seed_val)

	if MODEL_NAME == 'BERT':	
		tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
	elif MODEL_NAME == 'ROBERTA':
		tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

	sigmoid_fn = torch.nn.Sigmoid()

	if MODEL_NAME == 'BERT':
		# tree_path = './PT_PHEME5_FeatBERT40_Depth5_maxR5_MTL/'
		tree_path = './PT_PHEME5_FeatBERT40_Depth5_maxR5_MTL_modelSIT/'
	elif MODEL_NAME == 'ROBERTA':
		# tree_path = './PT_PHEME5_FeatROBERTA40_Depth5_maxR5_MTL/'
		tree_path = './PT_PHEME5_FeatROBERTA40_Depth5_maxR5_MTL_modelSIT/'

	
	files = ['charliehebdo.txt', 'germanwings-crash.txt', 'ottawashooting.txt','sydneysiege.txt']
	train_li,val_li = read_data(tree_path = tree_path,
								files = files,
								device = device)
	weight_vec, summ_weight_vec, pos_weight_vec, summ_pos_weight_vec = compute_weights(train_li = train_li, files = files, device = device)


	if TEST_FILE.startswith('charlie'):
		dftest = dfc.copy(deep=False)
	elif TEST_FILE.startswith('german'):
		dftest = dfg.copy(deep=False)
	elif TEST_FILE.startswith('ottawa'):
		dftest = dfo.copy(deep=False)
	else:
		dftest = dfs.copy(deep=False)


	for lr in lr_list:
		print("\n\n\nTraining with LR: ", lr)
		
		for test in files:
			if not test.startswith(TEST_FILE):
				continue
			
			# doubt
			# random.seed(seed_val)
			# numpy.random.seed(seed_val)
			# torch.manual_seed(seed_val)
			# torch.cuda.manual_seed(seed_val)		
					
			path = "./Models/"
			# path = "./drive/My Drive/IIT_Kgp/Research/Disaster/BTP_Chandana_Vishnu/verification/Models/"
			
			name = path + "mtl" + TEST_FILE + "_" + str(IN_FEATURES) + "_feat" + MODEL_NAME + ".pt"
			# print(args)
			# args, trainable_layers, in_features, out_features, classifier_dropout, mode
			model = MTLModel(args , TRAINABLE_LAYERS, IN_FEATURES, OUT_FEATURES, CLASSIFIER_DROPOUT)
			model.cuda(gpu_id)
			test_model = MTLModel(args , TRAINABLE_LAYERS, IN_FEATURES, OUT_FEATURES, CLASSIFIER_DROPOUT)
			test_model.cuda(gpu_id)
					
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
					test_trees.extend(train_li[filename])
					test_trees.extend(val_li[filename])
				else:
					curr_tree_dataset = TreeDataset(train_li[filename])
					train_trees.extend(curr_tree_dataset)
					val_data.extend(TreeDataset(val_li[filename]))
			
			print("size of training data", len(train_trees))
			print("Size of test data", len(test_trees))		
			print("\ntraining started....")
			
			prev_loss = math.inf
			prev_acc = 0		
			for i in range(NUM_ITERATIONS):
				
				model.train()			
				
				data_gen = DataLoader(
					train_trees,
					collate_fn=batch_tree_input,
					batch_size=BATCH_SIZE,
					shuffle = True
				)

				val_gen = DataLoader(
					val_data,
					collate_fn=batch_tree_input,
					batch_size=BATCH_SIZE,
					shuffle = True
				)
				
				ground_labels = []
				predicted_labels = []
				summ_ground_labels = []
				summ_predicted_labels = []
				# token_attentions = []
				j = 0
				train_avg_loss=0					
				err_count = 0
				for tree_batch in data_gen:
					loss, p_labels, p_probs, g_labels, p_summ_labels, p_summ_probs, summ_gt_labels , attentions = run(args = args, 
																													model = model, 
																													optimizer = optimizer, 
																													tree_batch = tree_batch, 
																													test_file = test_file, 
																													device=device,
																													weight_vec = weight_vec, 
																													pos_weight_vec = pos_weight_vec, 
																													summ_weight_vec = summ_weight_vec, 
																													summ_pos_weight_vec = summ_pos_weight_vec, 
																													mode="train")
					# token_attentions.extend(attentions)
					ground_labels.extend(g_labels)
					predicted_labels.extend(p_labels)
					summ_ground_labels.extend(summ_gt_labels)
					summ_predicted_labels.extend(p_summ_labels)
					j = j+1
					train_avg_loss += loss
				veri_train_acc = accuracy_score(ground_labels, predicted_labels)
				summ_train_acc = accuracy_score(summ_ground_labels, summ_predicted_labels)
				train_avg_loss /= j
				
				print("validation started..",len(val_data))
				model.eval()
				val_ground_labels = []
				val_predicted_labels= []
				val_summ_ground_labels = []
				val_summ_predicted_labels = []
				val_j = 0
				val_avg_loss = 0			
				with torch.no_grad():
					for batch in val_gen:
						loss, p_labels, p_probs, g_labels, p_summ_labels, p_summ_probs, summ_gt_labels, attentions = run(args = args, 
																													model = model, 
																													optimizer = optimizer, 
																													tree_batch = tree_batch, 
																													test_file = test_file, 
																													device=device,
																													weight_vec = weight_vec, 
																													pos_weight_vec = pos_weight_vec, 
																													summ_weight_vec = summ_weight_vec, 
																													summ_pos_weight_vec = summ_pos_weight_vec, 
																													mode="val")
						val_ground_labels.extend(g_labels)
						val_predicted_labels.extend(p_labels)
						val_summ_ground_labels.extend(summ_gt_labels)
						val_summ_predicted_labels.extend(p_summ_labels)
						val_j += 1
						val_avg_loss += loss			
				veri_val_acc = accuracy_score(val_ground_labels, val_predicted_labels)
				veri_val_f1 = f1_score(val_ground_labels, val_predicted_labels)
				summ_val_acc = accuracy_score(val_summ_ground_labels, val_summ_predicted_labels)
				summ_val_f1 = f1_score(val_summ_ground_labels, val_summ_predicted_labels)
				val_avg_loss /= val_j
				
				if MODEL_SAVING_POLICY == "acc":
					if(prev_acc <= veri_val_acc):
						save_model(model, optimizer, name, veri_val_acc, val_avg_loss)
						prev_acc = veri_val_acc
				else:			
					if(prev_loss >= val_avg_loss):
						save_model(model, optimizer, name, veri_val_acc, val_avg_loss)
						prev_loss = val_avg_loss
				
				print('Iteration ', i)
				print("errors ",err_count)			
				print('Training Loss: ', train_avg_loss)
				print('Verification Training accuracy: ', veri_train_acc)
				print('Summarization Training accuracy: ', summ_train_acc)
				print('Validation loss: ', val_avg_loss)			
				print('Verification Validation accuracy: ', veri_val_acc)
				print('Summarization Validation accuracy: ', summ_val_acc)
				print('Verification Validation f1 score: ', veri_val_f1)
				print('Summarization Validation f1 score: ', summ_val_f1)
				
				# scheduler.step(veri_val_acc)

				if (1 or (i+1) % 5 == 0 and i > 0):
					load_model(test_model, optimizer, name)
					print('Now Testing:', test_file)

					if L2_REGULARIZER == 'y':
						summary_filepath = "STL_" + MODEL_NAME + "_" + str(IN_FEATURES) + "_" + str(lr) + "_L2y_" + str(WEIGHT_DECAY) + '_' + TEST_FILE + "_summary.txt"
						pickle_path = "MTL_" + TEST_FILE + "_"  + MODEL_NAME + "_" + str(IN_FEATURES) + "_SIT_" + str(USE_SITU) + "_L2_y_" + str(WEIGHT_DECAY) + "_LR_" + str(lr) + "_LAM1_" + str(LAMBDA1) + "_LAM2_" + str(LAMBDA2) + ".pkl"
					else:
						summary_filepath = "STL_" + MODEL_NAME + "_" + str(IN_FEATURES) + "_" + str(lr) + "_L2n_" + TEST_FILE + "_summary.txt"
						pickle_path = "MTL_" + TEST_FILE + "_"  + MODEL_NAME + "_" + str(IN_FEATURES) + "_SIT_" + str(USE_SITU) + "_L2_n_LR_" + str(lr) + "_LAM1_" + str(LAMBDA1) + "_LAM2_" + str(LAMBDA2) + ".pkl"

					dfsum = testing( 
						test_model = test_model,
						tokenizer = tokenizer,
						dftest = dftest,
						test_trees = test_trees,
						device = device)
						
					create_summary(args = args,
						dfsum= dfsum,
						summary_filepath = summary_filepath,
						pickle_path = pickle_path)

	
			print('Training Complete')

			break
		break

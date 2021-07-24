import os
import sys
import math
import copy
import string
import codecs
import pickle
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, LoggingHandler
import rouge
import textstat
import gurobipy


sent_sim_model = SentenceTransformer('bert-base-nli-mean-tokens')


def get_numwords(x):
	tokens = [word for word in x['Orig_Tweet'].strip().split() if word not in string.punctuation]
	val1 = len(tokens)
	tokens = [word for word in x['Clean_Tweet'].strip().split() if word not in string.punctuation]
	val2 = len(tokens)
	tokens = [word for word in x['Norm_Tweet'].strip().split() if word not in string.punctuation]
	val3 = len(tokens)	
	val1 = 1 if math.isnan(val1) else val1
	val2 = 1 if math.isnan(val2) else val2
	val3 = 1 if math.isnan(val3) else val3
	return pd.Series([val1, val2, val3])


def generate_summary(Length, kappa):	
	predicted_summary_orig = []
	predicted_summary_clean = []
	total_verified = 0
	later_verified = 0
	summary_length = 0
	summary_length_orig = 0
	summary_length_clean = 0
	count = 0
	
	dfsum = pd.read_pickle('dfsum.pkl')
	dfpred = dfsum[dfsum['summ_pred']==1].reset_index(drop=True)
	dfpred = dfpred[dfpred['veri_pred']==0].reset_index(drop=True)
	dfpred['salience_score'] = (kappa * dfpred['summ_pred_prob']) + ((1 - kappa) * dfpred['veri_pred_prob'])
	
	if len(dfpred) != 0:
		dfpred.sort_values(by=['salience_score'], ascending=False, inplace=True)
		dfpred.drop_duplicates(subset=['Clean_Tweet'], keep='first', inplace=True)
		dfpred[["Num_words_orig", "Num_words_clean", "Num_words_norm"]] = dfpred.apply(lambda x : get_numwords(x), axis= 1)
		
		candidates = [] # (2-D list): each row contains [tweet_id, tweet, length in words, relevance_score]
		for i, row in dfpred.iterrows():
			candidates.append([row['Tweet_ID'], row['Clean_Tweet'], length, salience_score])

		# list of relevance scores of the tweets
		S = [i[3] for i in candidates]
		
		# list of lengths of tweets
		L = [i[2] for i in candidates]
		
		# list of tweets
		tweets = [i[1] for i in candidates]
		
		sentence_embeddings = sent_sim_model.encode(tweets)
		J = cosine_similarity(sentence_embeddings)
		for i in range(len(J)):
			for j in range(len(J[i])):			
				J[i][i] = 0
				if math.isnan(J[i][j]) or math.isinf(J[i][j]):
					print("Problem with similarity matrix..")			
		print("-------similarity of tweets obtained-----")
		assert len(candidates) == len(J), "Length of data and similarity matrix do not match.. Aborting.."
		
		
		###################################  Define the Model   ##################################
		
		m = gurobipy.Model("sol")

		# create variables

		#############################  Create a list of sentence variables  ######################

		X = m.addVars(len(candidates), lb=0, ub=1, vtype=gurobipy.GRB.BINARY, name='x')    
		
		#############################  Create a matrix of Y variables  ###########################
		
		# Y[i][j] represents a variable for each pair of sentence

		Y = m.addVars(len(candidates), len(candidates), lb=0, ub=1, vtype=gurobipy.GRB.BINARY, name='y')

		# update the variable environment
		m.update()

		######################################  Equations  #############################################

		# create the objective
		P = gurobipy.LinExpr() #contains objective function
		P = X.prod(S) - gurobipy.quicksum(Y[i, j] * J[i][j] for i in range(len(candidates)) for j in range(len(candidates)))
		# P = X.prod(S)
		
		# constraints to make the quadratic term in the objective function linear
		m.addConstrs(Y[i, j] >= X[i] + X[j] - 1 for i in range(len(candidates)) for j in range(len(candidates)))
		m.addConstrs(Y[i, j] <= (X[i] + X[j]) / 2.0 for i in range(len(candidates)) for j in range(len(candidates)))

		m.addConstr(X.prod(L) <= Length)
		
		m.setObjective(P, gurobipy.GRB.MAXIMIZE)

		########################  Solve the optimization problem and obtain summary  ###################

		try:
			m.optimize()
			for v in m.getVars():
				if v.x == 1:
					# print(v.varName)
					if 'x' in v.varName:
						count += 1
						sent_id = int(v.varName.split('[')[1].split(']')[0])
						tweet_id = candidates[sent_id][0]
						predicted_summary_orig.append(dfsum.loc[dfsum.Tweet_ID==tweet_id, 'Orig_Tweet'].values[0])
						predicted_summary_clean.append(dfsum.loc[dfsum.Tweet_ID==tweet_id, 'Clean_Tweet'].values[0])
						if int(dfsum.loc[dfsum.Tweet_ID==tweet_id, 'R1NR0'].values[0]) == 0:
							total_verified += 1
						elif int(dfsum.loc[dfsum.Tweet_ID==tweet_id, 'False0_True1_Unveri2_NR3_Rep4'].values[0]) == 0:
							later_verified += 1
						summary_length += int(dfsum.loc[dfsum.Tweet_ID==tweet_id, 'Num_words_norm'].values[0])
						summary_length_orig += int(dfsum.loc[dfsum.Tweet_ID==tweet_id, 'Num_words_orig'].values[0])
						summary_length_clean += int(dfsum.loc[dfsum.Tweet_ID==tweet_id, 'Num_words_clean'].values[0])						
						
		except gurobipy.GurobiError as e:
			print(e)
			sys.exit(0)

		summ_orig = '.\n'.join(predicted_summary_orig).strip()
		summ_clean = '.\n'.join(predicted_summary_clean).strip()
		
		if count == 0:
			veri_prop = 0
			modified_veri_prop = 0
		else:
			veri_prop = float(total_verified / count)
			modified_veri_prop = float((total_verified + later_verified) / count)
		
		return summ_orig, summ_clean, veri_prop, modified_veri_prop, summary_length, summary_length_orig, summary_length_clean, count, total_verified, later_verified


def prepare_results(p, r, f):
	return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


if __name__ == "__main__":
	for kappa in [0, 0.5, 1]:
		print(f'For kappa={kappa}')
		summ_orig, summ_clean, veri_prop, modified_veri_prop, summary_length, summary_length_orig, summary_length_clean, count, total_verified, later_verified = generate_summary(250, kappa)
		print(f'\nSummary generated for: {TEST_FILE}')
		print(f'Total tweets: {count}')
		print(f'Total verified: {total_verified}')
		print(f'Total later verified: {later_verified}')
		print(f'Summary length with normalized tweets: {summary_length}')
		print(f'Summary length with original tweets: {summary_length_orig}')
		print(f'Summary length with clean tweets: {summary_length_clean}')
		print(f'Verified_Ratio of tweets: {veri_prop}')
		print(f'Modified verified_Ratio of tweets: {modified_veri_prop}\n\n')

		# with open("../" + TEST_FILE + "_test_orig_250.txt",'w') as f:
		# 	f.write(summ_orig)

		# with open("../" + TEST_FILE + "_test_clean_250.txt",'w') as f:
		# 	f.write(summ_clean)


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
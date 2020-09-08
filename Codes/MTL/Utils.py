import random
import torch
import math
import pandas as pd

def save_model(model, optimizer, name, val_acc=0, val_loss=1):
	state = {
		'model':model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'val_acc': val_acc,
		'val_loss': val_loss
		}
	torch.save(state, name)


def load_model(model, optimizer, name):
	state = torch.load(name)
	model.load_state_dict(state['model'])
	optimizer.load_state_dict(state['optimizer'])
	print('Validation accuracy of the model is ', state.get('val_acc'))
	print('Validation loss of the model is ', state.get('val_loss'))
	return state.get('val_acc')


def get_numwords(x):
	val = len(x['Clean_Tweet'].strip().split())
	return 1 if math.isnan(val) else val


def generate_summary(numwords, dfsum):
	predicted_summary_orig = []
	predicted_summary_clean = []
	total_verified = 0
	summary_length = 0
	count = 0
	
	dfsum = dfsum[dfsum['Situational']==1].reset_index(drop=True)
	
	dfpred1 = dfsum[dfsum['summ_pred']==1].reset_index(drop=True)
	if len(dfpred1) != 0:
		dfpred1.sort_values(by=['summ_pred_prob'], ascending=False, inplace=True)
		dfpred1.drop_duplicates(subset=['Clean_Tweet'], keep='first', inplace=True)
		dfpred1["Num_words"] = dfpred1.apply(lambda x : get_numwords(x), axis= 1)
		for _, row in dfpred1.iterrows():
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
		dfpred0 = dfsum[dfsum['summ_pred']==0].reset_index(drop=True)
		if len(dfpred0) != 0:
			dfpred0.sort_values(by=['summ_pred_prob'], ascending=True, inplace=True)
			dfpred0.drop_duplicates(subset=['Clean_Tweet'], keep='first', inplace=True)
			dfpred0["Num_words"] = dfpred0.apply(lambda x : get_numwords(x), axis= 1)
			for _, row in dfpred0.iterrows():
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


def prepare_results(metric, p, r, f):
	return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def get_data(dftest, x):	
	orig_tweet_text = dftest.loc[dftest.Tweet_ID==int(x['Tweet_ID']), 'Orig_Tweet'].values[0]
	clean_tweet_text = dftest.loc[dftest.Tweet_ID==int(x['Tweet_ID']), 'Clean_Tweet'].values[0]
	sit_label = dftest.loc[dftest.Tweet_ID==int(x['Tweet_ID']), 'Situational'].values[0]
	orig_summ_gt = dftest.loc[dftest.Tweet_ID==int(x['Tweet_ID']), 'Summary_gt'].values[0]
	new_summ_gt = dftest.loc[dftest.Tweet_ID==int(x['Tweet_ID']), 'New_Summary_gt'].values[0]
	r1nr0 = dftest.loc[dftest.Tweet_ID==int(x['Tweet_ID']), 'R1NR0'].values[0]
	return pd.Series([orig_tweet_text, clean_tweet_text, sit_label, orig_summ_gt, new_summ_gt, r1nr0])

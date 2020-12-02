import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def get_stat(df):
	_veri = len(df[df.R1NR0 == 0])
	_unveri = len(df) - _veri
	_veri_situ = len(df[ (df.R1NR0 == 0) & (df.Situ1_NonSitu0_Oth2 == 1) ])
	_veri_nonSitu = _veri - _veri_situ
	_unveri_situ = len(df[ (df.R1NR0 == 1) & (df.Situ1_NonSitu0_Oth2 == 1) ])
	_unveri_nonSitu = _unveri - _unveri_situ
	print(f'\t\tVerified({_veri})\t\tUnverified({_unveri})\t\tTotal')
	print('Situ\tNon-Situ\t\tSitu\tNon-Situ')
	print(f'{_veri_situ}\t{_veri_nonSitu}\t\t{_unveri_situ}\t{_unveri_nonSitu}')

dfc = pd.read_pickle('data/features/charliehebdo_data.pkl')
dfc = dfc[dfc.SR == 'S'].reset_index(drop=True)
print('\ncharliehebdo:')
get_stat(dfc)
dfg = pd.read_pickle('data/features/germanwings-crash_data.pkl')
dfg = dfg[dfg.SR == 'S'].reset_index(drop=True)
print('\ngermanwings-crash:')
get_stat(dfg)
dfo = pd.read_pickle('data/features/ottawashooting_data.pkl')
dfo = dfo[dfo.SR == 'S'].reset_index(drop=True)
print('\nottawashooting:')
get_stat(dfo)
dfs = pd.read_pickle('data/features/sydneysiege_data.pkl')
dfs = dfs[dfs.SR == 'S'].reset_index(drop=True)
print('\nsydneysiege:')
get_stat(dfs)

dfc_gt = dfc[dfc.Summary_gt == 1]
print(f'Total ground truth summary tweets for charliehebdo:{len(dfc_gt)}')
dfg_gt = dfg[dfg.Summary_gt == 1]
print(f'Total ground truth summary tweets for germanwings-crash:{len(dfg_gt)}')
dfo_gt = dfo[dfo.Summary_gt == 1]
print(f'Total ground truth summary tweets for ottawashooting:{len(dfo_gt)}')
dfs_gt = dfs[dfs.Summary_gt == 1]
print(f'Total ground truth summary tweets for sydneysiege:{len(dfs_gt)}')

threshold = 0.7

def get_new_summ_gt_clean(row, event):
	if event == 'charliehebdo':
		tempdf = dfc_gt
	elif event == 'germanwings-crash':
		tempdf = dfg_gt
	elif event == 'ottawashooting':
		tempdf = dfo_gt
	else:
		tempdf = dfs_gt

	new_summ_gt_situ = 0
	new_summ_gt = 0
	if row['R1NR0'] == 0:
		for emb in tempdf['Clean_Emb']:
			score = cosine_similarity(row['Clean_Emb'].reshape(1, -1), emb.reshape(1, -1))[0][0]
			if score >= threshold:
				new_summ_gt = 1
				break

		if row['Situ1_NonSitu0_Oth2'] == 1:
			new_summ_gt_situ = new_summ_gt

	return pd.Series([new_summ_gt, new_summ_gt_situ])

print('\nAfter expansion:\n')
dfc[['New_Summ_gt_Clean', 'New_Summ_gt_Clean_Situ']] = dfc.apply(lambda x: get_new_summ_gt_clean(x, 'charliehebdo'), axis=1)
print('For charliehebdo:')
print(f'Ratio of New_Summ_gt_Clean: {len(dfc[dfc.New_Summ_gt_Clean == 1]) / len(dfc)}')
print(f'Ratio of New_Summ_gt_Clean_Situ: {len(dfc[dfc.New_Summ_gt_Clean_Situ == 1]) / len(dfc)}')

dfg[['New_Summ_gt_Clean', 'New_Summ_gt_Clean_Situ']] = dfg.apply(lambda x: get_new_summ_gt_clean(x, 'germanwings-crash'), axis=1)
print('For germanwings-crash:')
print(f'Ratio of New_Summ_gt_Clean: {len(dfg[dfg.New_Summ_gt_Clean == 1]) / len(dfg)}')
print(f'Ratio of New_Summ_gt_Clean_Situ: {len(dfg[dfg.New_Summ_gt_Clean_Situ == 1]) / len(dfg)}')

dfo[['New_Summ_gt_Clean', 'New_Summ_gt_Clean_Situ']] = dfo.apply(lambda x: get_new_summ_gt_clean(x, 'ottawashooting'), axis=1)
print('For ottawashooting:')
print(f'Ratio of New_Summ_gt_Clean: {len(dfo[dfo.New_Summ_gt_Clean == 1]) / len(dfo)}')
print(f'Ratio of New_Summ_gt_Clean_Situ: {len(dfo[dfo.New_Summ_gt_Clean_Situ == 1]) / len(dfo)}')

dfs[['New_Summ_gt_Clean', 'New_Summ_gt_Clean_Situ']] = dfs.apply(lambda x: get_new_summ_gt_clean(x, 'sydneysiege'), axis=1)
print('For sydneysiege:')
print(f'Ratio of New_Summ_gt_Clean: {len(dfs[dfs.New_Summ_gt_Clean == 1]) / len(dfs)}')
print(f'Ratio of New_Summ_gt_Clean_Situ: {len(dfs[dfs.New_Summ_gt_Clean_Situ == 1]) / len(dfs)}')


def get_new_summ_gt_bertweet(row, event):
	if event == 'charliehebdo':
		tempdf = dfc_gt
	elif event == 'germanwings-crash':
		tempdf = dfg_gt
	elif event == 'ottawashooting':
		tempdf = dfo_gt
	else:
		tempdf = dfs_gt

	new_summ_gt_situ = 0
	new_summ_gt = 0
	if row['R1NR0'] == 0:
		for emb in tempdf['Norm_Emb']:
			score = cosine_similarity(row['Norm_Emb'].reshape(1, -1), emb.reshape(1, -1))[0][0]
			if score >= threshold:
				new_summ_gt = 1
				break

		if row['Situ1_NonSitu0_Oth2'] == 1:
			new_summ_gt_situ = new_summ_gt

	return pd.Series([new_summ_gt, new_summ_gt_situ])

print('\nAfter expansion:\n')
dfc[['New_Summ_gt_BT', 'New_Summ_gt_BT_Situ']] = dfc.apply(lambda x: get_new_summ_gt_bertweet(x, 'charliehebdo'), axis=1)
print('For charliehebdo:')
print(f'Ratio of New_Summ_gt_BT: {len(dfc[dfc.New_Summ_gt_BT == 1]) / len(dfc)}')
print(f'Ratio of New_Summ_gt_BT_Situ: {len(dfc[dfc.New_Summ_gt_BT_Situ == 1]) / len(dfc)}')

dfg[['New_Summ_gt_BT', 'New_Summ_gt_BT_Situ']] = dfg.apply(lambda x: get_new_summ_gt_bertweet(x, 'germanwings-crash'), axis=1)
print('For germanwings-crash:')
print(f'Ratio of New_Summ_gt_BT: {len(dfg[dfg.New_Summ_gt_BT == 1]) / len(dfg)}')
print(f'Ratio of New_Summ_gt_BT_Situ: {len(dfg[dfg.New_Summ_gt_BT_Situ == 1]) / len(dfg)}')

dfo[['New_Summ_gt_BT', 'New_Summ_gt_BT_Situ']] = dfo.apply(lambda x: get_new_summ_gt_bertweet(x, 'ottawashooting'), axis=1)
print('For ottawashooting:')
print(f'Ratio of New_Summ_gt_BT: {len(dfo[dfo.New_Summ_gt_BT == 1]) / len(dfo)}')
print(f'Ratio of New_Summ_gt_BT_Situ: {len(dfo[dfo.New_Summ_gt_BT_Situ == 1]) / len(dfo)}')

dfs[['New_Summ_gt_BT', 'New_Summ_gt_BT_Situ']] = dfs.apply(lambda x: get_new_summ_gt_bertweet(x, 'sydneysiege'), axis=1)
print('For sydneysiege:')
print(f'Ratio of New_Summ_gt_BT: {len(dfs[dfs.New_Summ_gt_BT == 1]) / len(dfs)}')
print(f'Ratio of New_Summ_gt_BT_Situ: {len(dfs[dfs.New_Summ_gt_BT_Situ == 1]) / len(dfs)}')

dfc.drop(['Clean_Emb', 'Norm_Emb'], axis=1, inplace=True)
dfg.drop(['Clean_Emb', 'Norm_Emb'], axis=1, inplace=True)
dfo.drop(['Clean_Emb', 'Norm_Emb'], axis=1, inplace=True)
dfs.drop(['Clean_Emb', 'Norm_Emb'], axis=1, inplace=True)

# dfc.to_pickle('data/features/dfc_' + str(threshold) + '.pkl')
dfc.to_csv('data/features/dfc_labeled_final.tsv', sep='\t', index=False)
# dfg.to_pickle('data/features/dfg_' + str(threshold) + '.pkl')
dfg.to_csv('data/features/dfg_labeled_final.tsv', sep='\t', index=False)
# dfo.to_pickle('data/features/dfo_' + str(threshold) + '.pkl')
dfo.to_csv('data/features/dfo_labeled_final.tsv', sep='\t', index=False)
# dfs.to_pickle('data/features/dfs_' + str(threshold) + '.pkl')
dfs.to_csv('data/features/dfs_labeled_final.tsv', sep='\t', index=False)

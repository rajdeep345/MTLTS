# MTLVS: A Multi-Task Framework to Verify and Summarize Crisis-Related Microblogs

This repository contains codes and instructions for reproducing the results for our paper "MTLVS: A Multi-Task Framework to Verify and SummarizeCrisis-Related Microblogs".


------------------------------------------
## Folders
------------------------------------------

```
+ ./data/gt_summ - contains ground truth summary tweets 
+ ./data/features - contains the files required for creating the features and generating the discourse trees from tweet threads.
+ ./data/features/PT_PHEME5_FeatBERT40_Depth5_maxR5_MTL_Final - contains preprocessed trees with features extracted using BERT.
+ ./data/features/PT_PHEME5_FeatBERTWEET40_Depth5_maxR5_MTL_Final - contains preprocessed trees with features extracted using BERTweet.
+ ./data/summary_dataframes - contains the processed datasets with extended "in-summary" tweets after running Codes/expand_summ_gt.py.
+ ./Codes - contains codes for pre-processing the datasets and create features.
+ ./Codes/STLS - contains code for training summarization as a single task
+ ./Codes/STLV - contains code for training tweet verification as a single task, both with (stlv_final.py) and without (stlv_base.py) Tree-LSMTs. Script (grid_search_stlv.py) for performing exhaustive grid search with several hyper-parameter settings is also included.
+ ./Codes/MTLVS - contains code for generating verified summaries using Multi-Task Learning.
+ ./Codes/HMTLVS - contains code for generating verified summaries using Hierarchical Multi-Task Learning.
+ ./Codes/Analysis - contains codes for generating explanations using LIME (Lime_explanations.ipynb) and comparing the one dimensional verification loss curves of MTLVS and HMTLVS (oneD_loss_analysis.py).
```

------------------------------------------
## Dependencies
------------------------------------------
Please create the required conda environment using environment.yml
~~~
conda env create -f environment.yml
~~~

## **Dataset Preprocessing and Tree Generation**

Preprocessed discourse trees are already available under ./data/features as mentioned above. 
Hence Steps 1 - 5 may be skipped.

Step 1: 
Download pheme-rnr-dataset from https://figshare.com/articles/PHEME_dataset_of_rumours_and_non-rumours/4010619 and save it in ./data/pheme-rnr-dataset/.  
Download rumoureval2019 dataset from https://figshare.com/articles/RumourEval_2019_data/8845580 for stance labels and save it in ./data/rumoureval2019/. 

Step 2: Read data from pheme-rnr-dataset and ground truth summary labels
~~~
python ./Codes/create_data.py
~~~

Step 3: Expand summary ground truth labels as described in Section 3.1 of the paper and save the summary dataframes as pickle files.
~~~
python ./Codes/expand_summ_gt.py
~~~

Step 4: Create Features
Additional files required: 
  - slang.txt 
  - contractions.txt 
~~~
python ./Codes/create_features.py
~~~

Step 5: Generate Trees from the data  
Additional files required: 
  - summary pickle files present in ./data/summary_dataframes (output from Step 3)
  - output files from Step 4
  - all_tweets_posterior.txt
~~~
python ./Codes/generate_trees.py
~~~

## Training the Models
Step1: Download BERTweet(Bertweet_base_transformers) from https://github.com/VinAIResearch/BERTweet and save it under the root folder MTLVS. 

Step2: For training models using Leave-one-out principle and performing grid search  

**MTLVS**
~~~
python ./Codes/MTLVS/grid_search_mtl.py
~~~

**STLV**  
~~~
python ./Codes/STLV/grid_search_stlv.py
~~~
For training the model on *ferguson.txt*, the events arguments should be manually changed to 5 or has to be passed as command line argument.

Similar scripts can be written for taining HMTLVS(Codes/HMTLVS/hmtl4_final.py) and STLS(Codes/STLS/stl_summ.py) and STLV w/o TreeLSTM(Codes/STLS/stlv_base.py)

### Analysis
For generating 1-D Loss Plot for comparing MTLVS and HMTLVS
~~~
Set the following variables in oneD_loss_analysis.py
PLACE - place for which plot has to be generated
tree_path - path to the best MTL model 
path - path to the best HMTL model

python ./Codes/oneD_loss_analysis.py
~~~

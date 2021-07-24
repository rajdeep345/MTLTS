# MTLVS: A Multi-Task Framework to Verify and Summarize Crisis-Related Microblogs

This repository contains codes and instructions for reproducing the results for our paper "MTLVS: A Multi-Task Framework to Verify and Summarize Crisis-Related Microblogs".

## Data Folders

  - ./data/gt_summ - contains ground truth summary tweets for each event.
  - ./data/features - contains the files required for creating features and generating discourse trees from tweet threads.
  - ./data/features/PT_PHEME5_FeatBERT40_Depth5_maxR5_MTL_Final - contains preprocessed trees with features extracted using BERT.
  - ./data/features/PT_PHEME5_FeatBERTWEET40_Depth5_maxR5_MTL_Final - contains preprocessed trees with features extracted using BERTweet.
  - ./data/summary_dataframes - contains the processed datasets with extended "in-summary" tweets after running Codes/expand_summ_gt.py.

## Code Folders

  - ./Codes - contains codes for pre-processing the datasets, creating features, training the models, and performing content analysis of generated summaries.
  - ./Codes/Analysis - contains codes for analyzing MTLVS-generated summaries using WestClass and CatE.
  - ./Codes/models and ./Codes/utils - contain codes required for running SummaRuNNer-based stl_summ.py and mtlvs.py.
  - ./Codes/checkpoints and ./Codes/data - Auxilliary folders required for running the main codes.

## Dependencies

Please create the required conda environment using environment.yml
~~~
conda env create -f environment.yml
~~~

## Dataset Preprocessing and Tree Generation

Preprocessed discourse trees are already available under ./data/features as mentioned above. 
Hence Steps 1 - 5 may be skipped.

### Step 1: Download data ###

Download pheme-rnr-dataset from https://figshare.com/articles/PHEME_dataset_of_rumours_and_non-rumours/4010619 and save it in ./data/pheme-rnr-dataset/.  

Download rumoureval2019 dataset from https://figshare.com/articles/RumourEval_2019_data/8845580 for stance labels and save it in ./data/rumoureval2019/. 

### Step 2: Read data from pheme-rnr-dataset and ground truth summary labels ###
~~~
python ./Codes/create_data.py
~~~

### Step 3: Expand summary ground truth labels as described in Section 3.1 of the paper and save the summary dataframes as pickle files. ###
~~~
python ./Codes/expand_summ_gt.py
~~~

### Step 4: Create Features ###
Additional files required: 
  - ./Codes/slang.txt 
  - ./Codes/contractions.txt 
~~~
python ./Codes/create_features.py
~~~

### Step 5: Generate Trees from the data ###
Additional files required: 
  - summary pickle files present in ./data/summary_dataframes (output pickle files from Step 3).
  - output files from Step 4.
  - ./data/features/all_tweets_posterior.txt
~~~
python ./Codes/generate_trees.py
~~~


## Training the Models

Download BERTweet(Bertweet_base_transformers) from https://github.com/VinAIResearch/BERTweet and save it under the root folder MTLVS. 

### Train STLS - Summarization as a single task ###
~~~
python ./Codes/stl_summ.py [argument_list]
~~~
Instructions to run the code and sample outputs can be found in ``STLS_(BERT_Summarunner).ipynb``.

### Train STLV - Tweet Verification as a single task ###
Default values for various hyper-parameters are set in the code.
  
~~~
python ./Codes/stlv_final.py [argument_list]
~~~

We have included the script used to perform grid-search for hyper-parameter tuning for this task.
~~~
python ./Codes/grid_search_stlv.py
~~~

In order to reproduce the performance of STLV without Tree-LSTMs
~~~
python ./Codes/stlv_base.py [argument_list]
~~~

### Train MTLVS - Our proposed architecture to jointly train verification and summarization using Multi-task Learning ###
~~~
python ./Codes/mtlvs.py [argument_list]
~~~
Instructions to run the code and sample outputs can be found in ``mtlvs_setup.ipynb``.

``mtlvs.py`` saves a dataframe ``dfsum.pkl`` that contains all necessary information to generate the final summary including the tweet-level predictions from the <i>Summarizer</i> and <i>Verifier</i> modules.

Finally, we run ``ilp_summ.py`` that uses ``dfsum.pkl`` to generate the summary using <b>ILP</b> for various values of ``kappa``. It also calculates the summary statistics.
~~~
python ./Codes/ilp_summ.py
~~~

<!-- We have included the script to perform grid-search for hyper-parameter tuning for this task.
~~~
python ./Codes/grid_search_mtl.py
~~~ -->

## Analysis

Codes and instructions to analyze MTLVS-generated summaries using WestClass and CatE can be found under ``./Codes/Analysis``.

## Human Evaluation

### Survey form given to the participants (Identities of summaries hidden, summaries randomly placed in each section)
https://forms.gle/XqmpRzZgKmS7BAZk7

### For Reviewers: (Same survey form, Identities of summaries revealed)
https://forms.gle/GLS7sRRjgvamSdbv6

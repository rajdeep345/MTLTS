# MTLVS: A Multi-Task Framework to Verify and Summarize Crisis-Related Microblogs

This repository containes codes for the paper "MTLVS: A Multi-Task Framework to Verify and SummarizeCrisis-Related Microblogs".

Requirements

------------------------------------------
## Folders
------------------------------------------

```
+ ./Data - Contains the preprocessed data for Summary and Trees generated for Verification
+ ./Codes/MTLVS - Contains codes for model generating verified summaries using Multi-Task Learning 
+ ./Codes/HMTLVS - Contains codes for model generating verified summaries using Hierarchical Multi-Task Learning
+ ./Codes/STL - Contains codes for performing Single task verification and summarization
	-  STLV.py
	-  STLS.py
+ ./Codes/Analysis:
	-  LimeExplanations.py
	-  LimeWordClouds.py
	-  AttentionPlots.py
	-  1DLossPlot.py
```

------------------------------------------
## Models
------------------------------------------
~~~
Details of the Models implemented

Encoder - BERT/BERTweet

STL Verification:
* Encoder + TreeLSTM
* Encoder + FC
* Encoder + LSTM
* Fasttext + CNN - TextCNN

STL Summarization
* Encoder + FC

MTL - Verification + Summarization - Hard parameter sharing
Verification - Encoder + TreeLSTM
Summarization - Encoder + FC

Hierarchical MTL(HMTL) - Verification followed by Summarization
Encoder -> Verification Layers -> Summary Layers
~~~

------------------------------------------
## Dependencies
------------------------------------------
Pandas 1.1.1

------------------------------------------
## Instructions to run
------------------------------------------

### **Dataset Preprocessing and Tree Generation**

Step1: Download pheme-rnr-dataset from https://figshare.com/articles/PHEME_dataset_of_rumours_and_non-rumours/4010619 and save it in ./data/pheme-rnr-dataset/. Download rumoureval2019 data from https://figshare.com/articles/RumourEval_2019_data/8845580 for stance labels. Save it in ./data/rumoureval2019. 

Step2: Read data from pheme-rnr-dataset and ground truth summary labels
~~~
python create_data.py
~~~

Step3: Explan summary ground truth labels and store them in summary pickle files
~~~
python expand_summ_gt.py
~~~

Step4: Create Features from 
Additional files required - slang.txt, contractions.txt 
~~~
python create_features.py
~~~

Step5: Generate Trees from the data
Additional files required - summary pickle files present in ./data/summary_dataframes, output files from Step4 and all_tweets_posterior.txt
~~~
python generate_trees.py
~~~

### Training the Models
Step1: Download BERTweet(Bertweet_base_transformers) from https://github.com/VinAIResearch/BERTweet and save it under the root folder.

Step2: 
### Analysis
For generating 1-D Loss Plot for comparing MTLVS and HMTLVS
~~~
Set the following variables in oneD_loss_analysis.py
PLACE - place for which plot has to be generated
tree_path - path to the best MTL model 
path - path to the best HMTL model

python oneD_loss_analysis.py
~~~

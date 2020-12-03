# MTLVS: A Multi-Task Framework to Verify and Summarize Crisis-Related Microblogs

This repository containes codes for the paper "MTLVS: A Multi-Task Framework to Verify and SummarizeCrisis-Related Microblogs".

Requirements

------------------------------------------
*** Folders ***
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
*** Models ***
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
*** Dependencies ***
------------------------------------------
Pandas 1.1.1

------------------------------------------
*** Instructions to run ***
------------------------------------------
Dataset Preprocessing and Tree Generation
~~~
Step1: Create Features 
Additional files required - slang.txt, contractions.txt 
python create_features.py

Step2: Generate Trees 
Additional files required - summary pickle files present in ./data/summary_dataframes , output files from Step1
python generate_trees.py
~~~

For generating 1-D Loss Plot for comparing MTLVS and HMTLVS
~~~
PLACE - place for which plot has to be generated
tree_path - path to best MTL model 
path - path to best HMTL model
!python 1DLossPlot.py
~~~

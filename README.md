# MTLVS: A Multi-Task Framework to Verify and Summarize Crisis-Related Microblogs

This repository containes codes for the paper "MTLVS: A Multi-Task Framework to Verify and SummarizeCrisis-Related Microblogs".

Requirements

------------------------------------------
*** Folders ***
------------------------------------------

~~~
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
~~~

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


------------------------------------------
*** Instructions to run ***
------------------------------------------

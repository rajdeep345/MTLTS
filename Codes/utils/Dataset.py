import csv
import torch
import torch.utils.data as data
from torch.autograd import Variable
from .Vocab import Vocab
import numpy as np
import math, random
import pandas as pd

class Dataset(data.Dataset):
    def __init__(self, examples):
        super(Dataset,self).__init__()
        # data: {'sents':xxxx,'labels':'xxxx', 'summaries':[1,0]}
        self.examples = examples 
        self.training = False
    def train(self):
        self.training = True
        return self
    def test(self):
        self.training = False
        return self
    def shuffle(self,words):
        np.random.shuffle(words)
        return ' '.join(words)
    def dropout(self,words,p=0.3):
        l = len(words)
        drop_index = np.random.choice(l,int(l*p))
        keep_words = [words[i] for i in range(l) if i not in drop_index]
        return ' '.join(keep_words)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex
        #words = ex['sents'].split()
        #guess = np.random.random()

        #if self.training:
        #    if guess > 0.5:
        #        sents = self.dropout(words,p=0.3)
        #    else:
        #        sents = self.shuffle(words)
        #else:
        #    sents = ex['sents']
        #return {'id':ex['id'],'sents':sents,'labels':ex['labels']}
        
    def __len__(self):
        return len(self.examples)


class Place:
    def __init__(self,df,replicate=5):
        self.df = df
        self.replicate = replicate-1
        self.tweetidlist = random.sample(list(df.Tweet_ID), k=len(df))
    def __len__(self):
        return (self.replicate + 1)*len(self.df)

    def get(self, docsize):
        if not self.tweetidlist and not self.replicate:
            return None
            # raise Exception("Tweets exhausted")
        tweetidlist,self.tweetidlist = self.tweetidlist[:docsize],self.tweetidlist[docsize:]
        if not self.tweetidlist and self.replicate:
            self.tweetidlist = random.sample(list(self.df.Tweet_ID), k=len(self.df))
            self.replicate-=1
        df = self.df[self.df["Tweet_ID"].isin(tweetidlist)]
        df = df.sort_values("Date", ignore_index=True)
        return df
    
class Custom_Dataset(Dataset):
    def __init__(self, placelist, docsize):
        self.placelist = placelist
        self.docsize = docsize

    def __len__(self):
        s = sum([math.ceil(len(p)/self.docsize) for p in placelist])
        return s
    
    def __getitem__(self, index):
        place = random.choice(self.placelist)
        df = place.get(self.docsize)
        while df is None:
            self.placelist.remove(place)
            place = random.choice(self.placelist)
            df = place.get(self.docsize)
        tweetid = "\n".join(map(str,list(df.Tweet_ID)))
        doc = "\n".join(list(df.Clean_Tweet))
        labels = "\n".join(map(str,list(df.New_Summ_gt_Clean)))
        summaries = "None"
        # dfdoc = pd.DataFrame({'doc':doc,'labels':labels,'summaries':summaries})
        return (tweetid, doc, labels,summaries)

def createplacelist(traindflist, valdflist, testdf):
    maxlen = max([len(i) for i in traindflist])
    r = math.ceil(maxlen/64)
    trainplacelist = [Place(i,replicate=r) for i in traindflist]
    valplacelist = [Place(i,replicate=r) for i in valdflist]
    testplacelist = [Place(testdf, replicate=r)]

    return trainplacelist, valplacelist, testplacelist
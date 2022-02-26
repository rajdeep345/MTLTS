# SummaRuNNer with BERT utils/Vocab.py

import torch
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class Vocab():
    def __init__(self):
        pass
    def make_features(self,batch,sent_trunc=32,doc_trunc=100,split_token='\n'):
        sents_list,targets,doc_lens,tweet_ids = [],[],[],[]
        # trunc document
        # print(batch)
        for doc,label,tweetid in zip(batch['doc'],batch['labels'],batch['tweetid']):
            sents = doc.split(split_token)
            labels = label.split(split_token)
            tweetids = tweetid.split(split_token)
            labels = [int(l) for l in labels]
            max_sent_num = min(doc_trunc,len(sents))
            sents = sents[:max_sent_num]
            labels = labels[:max_sent_num]
            tweet_ids += tweetids[:max_sent_num]
            sents_list += sents
            targets += labels
            doc_lens.append(len(sents))

        input_ids = []
        attention_masks = []
        for doc in batch['doc']:
            doc_n = doc.split(split_token)
            k=0
            for sent in doc_n:
                encoded_dict = tokenizer.encode_plus(
                    sent,
                    None,
                    add_special_tokens=True,
                    max_length=32,
                    pad_to_max_length=True,
                    return_token_type_ids=True,
                    return_tensors="pt"
                    )
                k=k+1
                input_ids.append(encoded_dict['input_ids'])
                attention_masks.append(encoded_dict['attention_mask'])
                if(k==doc_trunc):
                    break
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        targets = torch.LongTensor(targets)
        summaries = batch['summaries']

        return input_ids,attention_masks,targets,summaries,doc_lens,tweet_ids

    def make_predict_features(self, batch, sent_trunc=150, doc_trunc=300, split_token='. '):
        sents_list, doc_lens = [],[]
        for doc in batch:
            sents = doc.split(split_token)
            max_sent_num = min(doc_trunc,len(sents))
            sents = sents[:max_sent_num]
            sents_list += sents
            doc_lens.append(len(sents))

        #features = torch.LongTensor(features)
        input_ids = []
        attention_masks = []
        for sent in sents_list:
            encoded_dict = tokenizer.encode_plus(
                    sent,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 128,           # Pad & truncate all sentences.
                    truncation = True,
                    pad_to_max_length = True,
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                 )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return  input_ids,attention_masks,doc_lens
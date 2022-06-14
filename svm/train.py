import json
from transformers import RobertaConfig, RobertaModel,RobertaTokenizer
import torch
from sklearn import svm, metrics
import numpy as np
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta = RobertaModel.from_pretrained('roberta-base')

def tokenize_sentence(sentence,  roberta, tokenizer,word_limits=25):
    tokens = tokenizer.encode(sentence)
    tokens = torch.tensor(tokens[:word_limits])
    return torch.reshape(roberta.embeddings.word_embeddings(tokens),(-1,)).detach().numpy()

def get_train_data(path,tokenizer,roberta, seq_len=0):
    #convs =[]
    data_samples = []
    gts = []
    gt_counter = {1:0,0:0}
    with open(path,'r') as f:
        for line in f.readlines():
            conv = json.loads(line)
            for utt in conv['utterances']:
                pkg_emb = []
                pkg = utt['pkg']
                pkg = [tuple(x) for x in pkg]
                pkg_ents = set()
                for (sub, pred, obj) in pkg:
                    pkg_ents.add(sub)
                    pkg_ents.add(obj)
                word_emb = tokenize_sentence(utt['text'],roberta,tokenizer)
                for rel in utt['relations']:
                    for em in [rel['head_span'],rel['child_span']]:
                        if not em['text'].lower() in ['i', 'my','he','his','her','they','their','our','we', 'she','hers']:
                            em_emb =tokenize_sentence(em['text'],roberta,tokenizer)
                            if em['personal_id'] in pkg_ents:
                                gt= 1
                                gt_counter[1] = gt_counter[1]+1
                            else:
                                gt = 0
                                gt_counter[0] = gt_counter[0]+1
                            input_emb =np.concatenate( (word_emb, em_emb,pkg_emb)).tolist()
                            data_samples.append(input_emb)
                            gts.append(gt)

                            seq_len = seq_len if seq_len > len(input_emb) else len(input_emb)
    temp = []
    for x in data_samples:
        pad = seq_len - len(x)
        x = x +[0]*pad
        temp.append(x)
    data_samples = temp
    els = gt_counter[1] if gt_counter[1] < gt_counter[0] else gt_counter[0]
    counter, temp_x,temp_gt = {0:0,1:0}, [],[]
    for x,y in zip(data_samples,gts):
        if counter[y] < els:
            temp_gt.append(y)
            counter[y] = counter[y]+1
            temp_x.append(x)
    data_samples,gts=temp_x,temp_gt

    return data_samples,gts, seq_len
train,train_gt,seq_len = get_train_data('/code2/data/input1.jsonl',tokenizer,roberta)
val,val_gt,_ = get_train_data('/code2/data/input2.jsonl',tokenizer,roberta,seq_len=seq_len)
test,test_gt,_ = get_train_data('/code2/data/input3.jsonl',tokenizer,roberta,seq_len=seq_len)

clf = svm.SVC(kernel='rbf')
clf.fit(train, train_gt)

train_pred = clf.predict(train)
val_pred = clf.predict(val)
test_pred = clf.predict(test)
print(f"Training: F1-score {metrics.f1_score(train_gt,train_pred)}, Precision {metrics.precision_score(train_gt,train_pred)}, Recall {metrics.recall_score(train_gt,train_pred)}\n")
print(f"Validation: F1-score {metrics.f1_score(val_gt,val_pred)}, Precision {metrics.precision_score(val_gt,val_pred)}, Recall {metrics.recall_score(val_gt,val_pred)}\n")
print(f"Test: F1-score {metrics.f1_score(test_gt,test_pred)}, Precision {metrics.precision_score(test_gt,test_pred)}, Recall {metrics.recall_score(test_gt,test_pred)}\n")
import pickle
pickle.dump(clf,open('svm.pickle','wb'))
#1.
#2. evenised
#3. without pronouns
#4. changed from 'linear' kernel to 'rbf' #linear good for text classification
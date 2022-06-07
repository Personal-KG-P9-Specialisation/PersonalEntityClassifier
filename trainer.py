from cgi import print_arguments
import os
import sys

import argparse
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from transformers import RobertaConfig, RobertaTokenizer
from torch.utils.data import DataLoader
import fitlog
from fastNLP import cache_results
from fastNLP import FitlogCallback, WarmupCallback, GradientClipCallback
from fastNLP import RandomSampler, TorchLoaderIter, LossInForward, Trainer, Tester
#from transformers import Trainer, TrainingArguments

from utils import MicroMetric
from model import CoLAKE
from dataset import PKGDataSet,PKGDatasetEvenDist


BATCH_SIZE = 16
EPOCHS=100
GRAD_ACCUMULATION=1
WARM_UP=0.1
LR=5e-5
BETA=0.999
WEIGHT_DECAY=0.01

#Model Hyperparameters
NUM_WORDS_URG = 100 
NUM_OBJS_URG = 10
NUM_RELS_URG = 10
N_PERS_ENTS = 10
N_PERS_CSKG = 10
N_PERS_RELS = 10
NUM_CSKG = 837
NUM_REL = 20
devices = list(range(torch.cuda.device_count()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = RobertaConfig.from_pretrained('roberta-base', type_vocab_size=6) #possibly 7
model = CoLAKE(config, NUM_WORDS_URG, NUM_OBJS_URG, NUM_RELS_URG, N_PERS_ENTS, N_PERS_CSKG, N_PERS_RELS, NUM_CSKG, NUM_REL)
model = model.to(device)
# fine-tune
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'embedding']
param_optimizer = list(model.named_parameters())
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': WEIGHT_DECAY},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr=LR, betas=(0.9, BETA), eps=1e-6)

metrics = [MicroMetric(pred='pred', target='target')]



#test_data_iter = TorchLoaderIter(dataset=test_set, batch_size=BATCH_SIZE, sampler=RandomSampler(),
#                                     num_workers=4)

#tester = Tester(data=test_data_iter, model=model, metrics=metrics, device=devices)
    # tester.test()

#fitlog_callback = FitlogCallback(tester=tester, log_loss_every=100, verbose=1)
gradient_clip_callback = GradientClipCallback(clip_value=1, clip_type='norm')
warmup_callback = WarmupCallback(warmup=WARM_UP, schedule='linear')

bsz = BATCH_SIZE // GRAD_ACCUMULATION
train_set = PKGDatasetEvenDist('data/input1.jsonl')
dev_set = PKGDataSet('data/input2.jsonl')
test_set = PKGDataSet('data/input3.jsonl')
train_data_iter = TorchLoaderIter(dataset=train_set,
                                      batch_size=bsz,
                                      collate_fn=train_set.collate_fn)
dev_data_iter = TorchLoaderIter(dataset=dev_set,
                                    batch_size=bsz,
                                    collate_fn=dev_set.collate_fn)
test_iter = TorchLoaderIter(dataset=test_set,
                                    batch_size=bsz,
                                    collate_fn=dev_set.collate_fn)
#args = TrainingArguments('test',do_train=True)
#trainer = Trainer(model=model,train_dataset=train_set,eval_dataset=train_set,args=args)
if len(sys.argv) >=2 and str(sys.argv[1]) == 'gpu':
    trainer =Trainer(train_data=train_data_iter,
                      dev_data=dev_data_iter,
                      model=model,
                      optimizer=optimizer,
                      loss=LossInForward(),
                      batch_size=bsz,
                      update_every=GRAD_ACCUMULATION,
                      n_epochs=EPOCHS,
                      metrics=metrics,
                      callbacks=[gradient_clip_callback, warmup_callback],
                      #callbacks=[fitlog_callback, gradient_clip_callback, warmup_callback],
                      device=device,
                      use_tqdm=True)
else:
    trainer = Trainer(train_data=train_data_iter,
                      dev_data=dev_data_iter,
                      model=model,
                      optimizer=optimizer,
                      loss=LossInForward(),
                      batch_size=bsz,
                      update_every=GRAD_ACCUMULATION,
                      n_epochs=EPOCHS,
                      metrics=metrics,
                      callbacks=[gradient_clip_callback, warmup_callback],
                      #callbacks=[fitlog_callback, gradient_clip_callback, warmup_callback],
                      #device=devices,
                      use_tqdm=True)

trainer.train()
if len(sys.argv) >=2 and str(sys.argv[1]) == 'gpu': 
    tester = Tester(data=test_iter, model=model, metrics=metrics, device=devices)
else:
    tester = Tester(data=test_iter, model=model, metrics=metrics)
tester.test()


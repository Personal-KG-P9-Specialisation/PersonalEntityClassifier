import os
import sys

import argparse
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from transformers import RobertaConfig, RobertaTokenizer

import fitlog
from fastNLP import cache_results
from fastNLP import FitlogCallback, WarmupCallback, GradientClipCallback
from fastNLP import RandomSampler, TorchLoaderIter, LossInForward, Trainer, Tester


from utils import MicroMetric
from model import CoLAKE
from dataset import PKGDataSet


BATCH_SIZE = 32
EPOCHS=100
GRAD_ACCUMULATION=1
WARM_UP=0.1
LR=5e-5
BETA=0.999
WEIGHT_DECAY=0.01

config = RobertaConfig.from_pretrained('roberta-base', type_vocab_size=6) #possibly 7
model = CoLAKE(config, )

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



test_data_iter = TorchLoaderIter(dataset=test_set, batch_size=BATCH_SIZE, sampler=RandomSampler(),
                                     num_workers=4)
devices = list(range(torch.cuda.device_count()))
tester = Tester(data=test_data_iter, model=model, metrics=metrics, device=devices)
    # tester.test()

fitlog_callback = FitlogCallback(tester=tester, log_loss_every=100, verbose=1)
gradient_clip_callback = GradientClipCallback(clip_value=1, clip_type='norm')
warmup_callback = WarmupCallback(warmup=WARM_UP, schedule='linear')

bsz = BATCH_SIZE // GRAD_ACCUMULATION

train_data_iter = TorchLoaderIter(dataset=train_set,
                                      batch_size=bsz,
                                      sampler=RandomSampler(),
                                      num_workers=4)
dev_data_iter = TorchLoaderIter(dataset=dev_set,
                                    batch_size=bsz,
                                    sampler=RandomSampler(),
                                    num_workers=4)
trainer = Trainer(train_data=train_data_iter,
                      dev_data=dev_data_iter,
                      model=model,
                      optimizer=optimizer,
                      loss=LossInForward(),
                      batch_size=bsz,
                      update_every=GRAD_ACCUMULATION,
                      n_epochs=EPOCHS,
                      metrics=metrics,
                      callbacks=[fitlog_callback, gradient_clip_callback, warmup_callback],
                      device=devices,
                      use_tqdm=True)

trainer.train(load_best_model=False)


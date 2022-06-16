from cgi import print_arguments
import sys,os
import time
import torch
from torch import optim
from transformers import RobertaConfig
from fastNLP import RandomSampler, TorchLoaderIter, LossInForward, Trainer, Tester,cache_results,FitlogCallback, WarmupCallback, GradientClipCallback
#from transformers import Trainer, TrainingArguments

from utils import MicroMetric,LossMetric,PrecisionSigmoidMetric
from model import PEC, PEC_Sig
from dataset import PKGDataSet,PKGDatasetEvenDist,PKGDatasetSig
import numpy as np
np.random.seed(101)
#For Sckit learn, which it handles correctly.
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.getenv('output')
if not os.path.exists(OUTPUT_DIR):
    os.system(f'mkdir {OUTPUT_DIR}')


BATCH_SIZE = 16
EPOCHS=200
GRAD_ACCUMULATION=1
WARM_UP=0.1
LR=5e-5#1e-4#5e-5
BETA=0.999
WEIGHT_DECAY=0.01

#Model Hyperparameters
NUM_WORDS_URG = 24 
NUM_OBJS_URG = 19#18
NUM_RELS_URG = 5#,4
N_PERS_ENTS = 24#20
N_PERS_CSKG = 16
N_PERS_RELS = 42#,36

NUM_CSKG = 837
NUM_REL = 20

devices = list(range(torch.cuda.device_count()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = RobertaConfig.from_pretrained('roberta-base', type_vocab_size=6) #possibly 7
model = PEC_Sig(config, NUM_CSKG, NUM_REL)
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

metrics = [PrecisionSigmoidMetric(pred='pred', target='target'),LossMetric(loss='loss')]



#test_data_iter = TorchLoaderIter(dataset=test_set, batch_size=BATCH_SIZE, sampler=RandomSampler(),
#                                     num_workers=4)

#tester = Tester(data=test_data_iter, model=model, metrics=metrics, device=devices)
    # tester.test()

#fitlog_callback = FitlogCallback(tester=tester, log_loss_every=100, verbose=1)
gradient_clip_callback = GradientClipCallback(clip_value=1, clip_type='norm')
warmup_callback = WarmupCallback(warmup=WARM_UP, schedule='linear')

bsz = BATCH_SIZE // GRAD_ACCUMULATION
train_set = PKGDatasetSig('data/input1.jsonl',max_pkg_ents=N_PERS_ENTS, max_pkg_rels=N_PERS_RELS, max_pkg_cskg=N_PERS_CSKG, max_words=NUM_WORDS_URG, max_rels=NUM_RELS_URG, max_tails=NUM_OBJS_URG)
dev_set = PKGDatasetSig('data/input2.jsonl',max_pkg_ents=N_PERS_ENTS, max_pkg_rels=N_PERS_RELS, max_pkg_cskg=N_PERS_CSKG, max_words=NUM_WORDS_URG, max_rels=NUM_RELS_URG, max_tails=NUM_OBJS_URG)
test_set = PKGDatasetSig('data/input3.jsonl',max_pkg_ents=N_PERS_ENTS, max_pkg_rels=N_PERS_RELS, max_pkg_cskg=N_PERS_CSKG, max_words=NUM_WORDS_URG, max_rels=NUM_RELS_URG, max_tails=NUM_OBJS_URG)
train_data_iter = TorchLoaderIter(dataset=train_set,
                                      batch_size=bsz,
                                      collate_fn=train_set.collate_fn, sampler=RandomSampler())

#Temp code to find max_pkg_ent,etc.
"""n_pkg_ents,n_pkg_cskg,n_pkg_rels,n_word_nodes,n_relation_nodes,n_tail_mentions = 0,0,0,0,0,0
for i,_ in train_data_iter.dataiter:
    n_pkg_ents = i['n_pkg_ents'][0] if i['n_pkg_ents'][0] > n_pkg_ents else n_pkg_ents
    n_pkg_rels = i['n_pkg_rels'][0] if i['n_pkg_rels'][0] > n_pkg_rels else n_pkg_rels
    n_pkg_cskg =i['n_pkg_cskg'][0] if i['n_pkg_cskg'][0] > n_pkg_cskg else n_pkg_cskg
    n_word_nodes = i['n_word_nodes'][0] if i['n_word_nodes'][0] > n_word_nodes else n_word_nodes
    n_relation_nodes = i['n_relation_nodes'][0] if i['n_relation_nodes'][0] > n_relation_nodes else n_relation_nodes
    n_tail_mentions = i['n_tail_mentions'][0] if i['n_tail_mentions'][0] > n_tail_mentions else n_tail_mentions
    continue
print(f'n_pkg_ents: {n_pkg_ents}, n_pkg_rels: {n_pkg_rels}, n_pkg_cskg: {n_pkg_cskg}, n_word_nodes: {n_word_nodes}')
print(f'n_relation_nodes: {n_relation_nodes}, n_tail_mentions: {n_tail_mentions}')"""
dev_data_iter = TorchLoaderIter(dataset=dev_set,
                                    batch_size=bsz,
                                    collate_fn=dev_set.collate_fn, sampler=RandomSampler())
test_iter = TorchLoaderIter(dataset=test_set,
                                    batch_size=bsz,
                                    collate_fn=dev_set.collate_fn, sampler=RandomSampler())

#test code
"""for i,y in train_data_iter.dataiter:
    p = model(i['input_ids'],i['attention_mask'],i['token_type_ids'],i['position_ids'],None, None,i['n_pkg_ents'],i['n_pkg_cskg'],i['n_pkg_rels'],i['n_word_nodes'],i['n_relation_nodes'],i['n_tail_mentions'],i['target'])
    print(p)
    exit()"""

if len(sys.argv) >=2 and str(sys.argv[1]) == 'gpu':
    trainer =Trainer(train_data=train_data_iter,
                      dev_data=dev_data_iter,
                      model=model,
                      optimizer=optimizer,
                      save_path=OUTPUT_DIR,
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
                      save_path=OUTPUT_DIR,
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
start = time.time()
trainer.train()
print(f"training finished in {time.time()-start}s")
#assumes only one model
model = torch.load(f'{OUTPUT_DIR}/'+os.listdir(f'{OUTPUT_DIR}')[0])
if len(sys.argv) >=2 and str(sys.argv[1]) == 'gpu': 
    tester = Tester(data=test_iter, model=model, metrics=metrics, device=devices)
else:
    tester = Tester(data=test_iter, model=model, metrics=metrics)
tester.test()

print('train_data results on best performing model')

if len(sys.argv) >=2 and str(sys.argv[1]) == 'gpu': 
    tester = Tester(data=test_iter, model=model, metrics=metrics, device=devices)
else:
    tester = Tester(data=train_data_iter, model=model, metrics=metrics)
tester.test()

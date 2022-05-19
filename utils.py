import torch
from fastNLP.core.metrics import MetricBase
from fastNLP.core.utils import _get_func_signature
from sklearn.metrics import f1_score, precision_recall_fscore_support

class MicroMetric(MetricBase):
    def __init__(self, pred=None, target=None, no_relation_idx=0):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=None)
        self.no_relation = no_relation_idx
        self.num_predict = 0
        self.num_golden = 0
        self.true_positive = 0

    def evaluate(self, pred, target, seq_len=None):
        '''
        :param pred: batch_size
        :param target: batch_size
        :param seq_len: not uesed when doing text classification
        :return:
        '''

        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if pred.dim() != target.dim():
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        preds = pred.detach().cpu().numpy().tolist()
        targets = target.to('cpu').numpy().tolist()
        for pred, target in zip(preds, targets):
            if pred == target and pred != self.no_relation:
                self.true_positive += 1
            if target != self.no_relation:
                self.num_golden += 1
            if pred != self.no_relation:
                self.num_predict += 1

    def get_metric(self, reset=True):
        if self.num_predict > 0:
            micro_precision = self.true_positive / self.num_predict
        else:
            micro_precision = 0.
        micro_recall = self.true_positive / self.num_golden
        micro_fscore = self._calculate_f1(micro_precision, micro_recall)
        evaluate_result = {
            'f_score': micro_fscore,
            'precision': micro_precision,
            'recall': micro_recall
        }

        if reset:
            self.num_predict = 0
            self.num_golden = 0
            self.true_positive = 0

        return evaluate_result

    def _calculate_f1(self, p, r):
        if r == 0.:
            return 0.
        return 2 * p * r / float(p + r)
data= None
import json, networkx as nx, numpy as np
from transformers import RobertaTokenizer
def create_input_data_file(input_file, max_utt_Len):
    global data
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    convs = []
    with open(input_file,'r') as f:
        for line in f.readlines():
            convs.append(json.loads(line))
            break
    for conv in convs:
        for utt in conv['utterances']:    
            if not utt['relations'] == []:
                data = create_input(tokenizer, utt,max_utt_Len)
                break

def create_input(tokenizer, utt, max_utt_len):
    #-------------------------------------------------------------
    ## Utterance Relation Graph
    anchor_words=[]
    rel_map={}
    for x in utt['relations']:
        anchor_words.append(utt['text'][x['head_span']['start']:x['head_span']['end']])
        rel_map[utt['text'][x['head_span']['start']:x['head_span']['end']]] = x
        #anchor_words.append(utt['text'][x['child_span']['start']:x['child_span']['end']])
        #print(x)
    #words = tokenizer.encode( utt['text'])
    #words = words[:max_utt_len]
    text = utt['text']
    text = text.replace("."," ").replace("?"," ")..replace("!"," ").replace(","," ")
    words = text.split(' ')
    tokens,relations, tail_words = [0],[],[]
    node2label = {0: 0}
    idx = 1
    pos = 0
    soft_position = [0]
    relation_position=[]
    tail_position=[]
    em = False
    em_indices = []
    #print("\n\n",rel_map.keys())
    for word in words:
        if len(word) <= 0:
            continue
        if word in anchor_words:
            em=True
            
            relations.append((idx,rel_map[word]['label']))
            relation_position.append(pos)
            tail_words.append(utt['text'][rel_map[word]['child_span']['start']:rel_map[word]['child_span']['end']])
            
            #pos +=1
        
        word_enc = tokenizer.encode(word, add_special_tokens=False, add_prefix_space=True)
        word_enc = word_enc[:max_utt_len]
        for w in word_enc:
            node2label[idx] = w
            tokens.append(w)
            soft_position.append(pos)
            if em:
                em_indices.append(idx)
            idx +=1
            pos += 1
        
        em=False
        if len(relations) == 0:
            continue
    node2label[idx] =2
    tokens.append(2)
    soft_position.append(pos)
    idx +=1
    assert len(tokens) == idx
    G = nx.complete_graph(idx)
    n_word_nodes = idx
        
    for rel, pos_r in zip(relations,relation_position):
        idx +=1
        rel_idx,rel_ac = rel
            #if rel_ac not in G.nodes:
        G.add_node(idx)
        node2label[idx] = rel_ac
        soft_position.append(pos_r+1)
        G.add_edge(rel_idx,idx)
    n_relation_nodes = idx - n_word_nodes
        
    for tail, pos_t in zip(tail_words,relation_position):
            #if tail not in G.nodes:
        tail_enc = tokenizer.encode(tail)
        posi = pos_t+2
        prev = None
        for i in tail_enc:
            idx += 1
            G.add_node(idx)
            if prev is None:
                prev = idx
            else:
                G.add_edge(prev,idx)
                prev = idx
            node2label[idx] = i
            soft_position.append(posi)
            posi += 1
    n_tail_mentions = idx - n_relation_nodes-n_word_nodes
        
    #-------------------------------------------------------------
    ##Personal Knowledge Graph Encoding
    pkg = utt['pkg']
    pkg_ent = []
    pred_obj_map = {}
    for (sub, pred, obj) in pkg:
        idx +=1
        node2label[idx] = sub
        pred_obj_map[idx] = (pred,obj)
    


    #-------------------------------------------------------------
    #0 = words, 1 = relations, 2 = entity mentions
    token_types = [0] * n_word_nodes + [1] * n_relation_nodes +[2] * n_tail_mentions
    for i in em_indices:
        token_types[i+1] = 2
    adj = np.array(nx.adjacency_matrix(G).todense())
    adj = adj + np.eye(adj.shape[0], dtype=int)

    return {'n_word_nodes': n_word_nodes, 'n_relation_nodes': n_relation_nodes,'n_object_nodes':n_tail_mentions, 'nodes': [node2label[k] for k in G.nodes],
                                      'soft_position': soft_position, 'adj': adj.tolist(),
                                      'token_type_ids': token_types}
create_input_data_file("data/total_dataset.jsonl", 10000)
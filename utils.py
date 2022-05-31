import torch
from fastNLP.core.metrics import MetricBase
from fastNLP.core.utils import _get_func_signature
from sklearn.metrics import f1_score, precision_recall_fscore_support
import itertools
import matplotlib.pyplot as plt
import queue

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
def create_input_data_file(input_file, max_utt_Len, output_file):
    #global data
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    convs = []
    with open(input_file,'r') as f:
        for line in f.readlines():
            convs.append(json.loads(line))
    for conv in convs:
        for utt in conv['utterances']:    
            if not utt['relations'] == []:
                data = create_input(tokenizer, utt,max_utt_Len)
                utt['input_pec'] = data
    with open(output_file, 'w') as f:
        for conv in convs:
            f.write(json.dumps(conv)+'\n')

def create_complete_sub_G(G,lst, new_nodes=True):
    edges = itertools.combinations(lst,2)
    if new_nodes:
        G.add_nodes_from(lst)
    for x in lst:
        for y in lst:
            G.add_edge(x,y)
    return G

def is_int(x):
    try:
        int(x)
        return True
    except ValueError:
        return False

def add_edge_rel_tail(G, rel_idx, tail_indices):
    for x in tail_indices:
        G.add_edge(rel_idx,x)
    return G



def create_input(tokenizer, utt, max_utt_len):
    
    def split_text(text):
        splits = text.split(' ')
        indices = []
        sofar = 0
        for x in splits:
            indices.append((sofar,sofar+len(x)))
            sofar+= 1
            sofar+= len(x)
        return splits,indices
    
    tokens = [0]
    node2label = {0: 0}
    idx = 1
    pos = 0
    soft_position = [0]
    n_pkg_ents,n_pkg_rels = 0,0
    #-------------------------------------------------------------
    ##Personal Knowledge Graph Encoding
    CSKG_type = []
    if 'pkg' in utt.keys():
        pkg = utt['pkg']
        pkg = [tuple(x) for x in pkg]
        pkg_ent = set()
        pkg_ent2idx = {}
        preds = {}


        for (sub, pred, obj) in pkg:
            pkg_ent.add(sub)
            pkg_ent.add(obj)
            #preds[(sub,obj,pred)] = pred
        
        pkg_ents = [str(x) for x in pkg_ent]
        pkg_ents = sorted(list(pkg_ents))
        for x in pkg_ents:
            if is_int(x):
                x = int(x)
        for ent in pkg_ents:
            
            node2label[idx] = ent
            tokens.append(ent)
            soft_position.append(pos)

            if not is_int(ent):
                CSKG_type.append(idx)
            pkg_ent2idx[ent] = idx
            idx +=1
            pos += 1
        n_pkg_ents = idx
        G = nx.Graph()
        G.add_nodes_from([x for x in range(idx)])
        for (sub,pred,obj) in pkg:
            node2label[idx] = pred
            tokens.append(pred)
            soft_position.append(pos)
            G.add_node(idx)
            G.add_edge(pkg_ent2idx[str(sub)],idx)
            G.add_edge(pkg_ent2idx[str(obj)],idx)

            idx += 1
            pos += 1
        n_pkg_rels = idx - n_pkg_ents
        

    #-------------------------------------------------------------
    ## Utterance Relation Graph
    anchor_words,anchor_indices, relations, tail_words,rel_map = [],[], [],[],{}
    personal_ids, temp_personal_ids = [],[]

    for x in utt['relations']:
        anchor_words.append((utt['text'][x['head_span']['start']:x['head_span']['end']]))
        anchor_indices.append((x['head_span']['start'],x['head_span']['end']))
        temp_personal_ids.append(x['head_span']['personal_id'])
        rel_map[(utt['text'][x['head_span']['start']:x['head_span']['end']],(x['head_span']['start'],x['head_span']['end']))] = x
        
        """if utt['text'][x['head_span']['start']:x['head_span']['end']] in rel_map.keys():
            rel_map[utt['text'][x['head_span']['start']:x['head_span']['end']]].put(x)
        else:
            rel_map[utt['text'][x['head_span']['start']:x['head_span']['end']]] = queue.Queue()
            rel_map[utt['text'][x['head_span']['start']:x['head_span']['end']]].put(x)"""
        #anchor_words.append(utt['text'][x['child_span']['start']:x['child_span']['end']])
        #print(x)
    #words = tokenizer.encode( utt['text'])
    #words = words[:max_utt_len]
    
    text = utt['text']
    text = text.replace("."," ").replace("?"," ").replace("!"," ").replace(","," ")
    words, word_indices = split_text(text)
    
    relation_position=[]
    em, em_indices, em_count = False,[],0
    utt_len = 0
    for word, word_index in zip(words,word_indices):
        if len(word) <= 0:
            continue
        if word in anchor_words and word_index in anchor_indices:
            em=True
            places = len(tokenizer.encode(word, add_special_tokens=False, add_prefix_space=True)[:max_utt_len])
            personal_ids.extend([temp_personal_ids[em_count]]*places)
            em_count += 1

            relations.append((idx,places,rel_map[word,word_index]['label']))
            relation_position.append(pos)
            tail_words.append((len(relations)-1,utt['text'][rel_map[word,word_index]['child_span']['start']:rel_map[word,word_index]['child_span']['end']]))
            temp_personal_ids.append(rel_map[word,word_index]['child_span']['personal_id'])
            
        
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
        utt_len += len(word)
        if len(relations) == 0:
            continue
    assert len(personal_ids) == len(em_indices)
    node2label[idx] =2
    tokens.append(2)
    soft_position.append(pos)
    idx +=1
    assert len(tokens) == idx
    
    n_word_nodes = idx - (n_pkg_ents +n_pkg_rels)
    if 'pkg' in utt.keys():
        G = create_complete_sub_G(G , [x for x in range(idx-n_word_nodes, idx)])
    else:
        G = nx.complete_graph(idx)
    
    rel2idx = []    
    for rel, pos_r in zip(relations,relation_position):
        idx +=1
        rel_idx,span,rel_ac = rel
            #if rel_ac not in G.nodes:
        G.add_node(idx)
        node2label[idx] = rel_ac
        soft_position.append(pos_r+1)
        #G.add_edge(rel_idx,idx)
        for x in range(span):
            G.add_edge(rel_idx+x, idx)
        rel2idx.append(idx)
    n_relation_nodes = idx - (n_word_nodes+n_pkg_ents +n_pkg_rels)

    all_tail_indices = []
    for (rel_idx,tail), pos_t in zip(tail_words,relation_position):
            #if tail not in G.nodes:
        tail_indices = []
        tail_enc = tokenizer.encode(tail)
        posi = pos_t+2
        prev = None
        for i in tail_enc:
            idx += 1
            G.add_node(idx)
            tail_indices.append(idx)
            """if prev is None:
                prev = idx
            else:
                G.add_edge(prev,idx)
                prev = idx"""
            node2label[idx] = i
            soft_position.append(posi)
            posi += 1
        G = create_complete_sub_G(G,tail_indices, new_nodes=False)
        G = add_edge_rel_tail(G, rel2idx[rel_idx], tail_indices)
        personal_ids.extend([temp_personal_ids[em_count]]*len(tail_indices))
        all_tail_indices.extend(tail_indices)
        em_count += 1
    n_tail_mentions = idx - (n_relation_nodes+n_word_nodes+n_pkg_ents +n_pkg_rels)
    #nx.draw(G, pos=nx.spring_layout(G))
    #plt.savefig('test.png')

    #-------------------------------------------------------------
    #0 = words, 1 = relations, 2 = entity mentions, 3 = pkg_ents, 4 = CSKG ent
    if 'pkg' in utt.keys():
        token_types = [0]+[3]*(n_pkg_ents-1)+[1]*n_pkg_rels +  [0] * n_word_nodes + [1] * n_relation_nodes +[2] * n_tail_mentions
    else:
        token_types = [3]*(n_pkg_ents-1)+[1]*n_pkg_rels +  [0] * n_word_nodes + [1] * n_relation_nodes +[2] * n_tail_mentions
    for i in em_indices:
        token_types[i] = 2
    for i in CSKG_type:
        token_types[i] = 4
    adj = np.array(nx.adjacency_matrix(G).todense())
    adj = adj + np.eye(adj.shape[0], dtype=int)
    nodes = [node2label[k] for k in G.nodes]
    assert len(nodes) == n_word_nodes + n_relation_nodes + n_tail_mentions + n_pkg_ents + n_pkg_rels
    assert len(soft_position) == n_word_nodes+n_relation_nodes+n_tail_mentions+ n_pkg_ents + n_pkg_rels
    assert len(token_types) == n_word_nodes+n_relation_nodes+n_tail_mentions+ n_pkg_ents + n_pkg_rels
    em_indices.extend(all_tail_indices)
    assert len(personal_ids) == len(em_indices)
    return {'n_pkg_ents':n_pkg_ents,'n_pkg_rels':n_pkg_rels, 'n_word_nodes': n_word_nodes, 'n_relation_nodes': n_relation_nodes,'n_object_nodes':n_tail_mentions, 'nodes': nodes,
                                      'soft_position': soft_position, 'adj': adj.tolist(),
                                      'token_type_ids': token_types, 'entity_mention_idx':(em_indices,personal_ids)}
if __name__ == '__main__':
    create_input_data_file("data/total_dataset.jsonl", 10000, "data/pec_convs.jsonl")
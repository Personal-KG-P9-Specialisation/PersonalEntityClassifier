
from re import X
from torch.utils.data import Dataset
import json
import torch

class PKGDataSet(Dataset):
    def __init__(self, path):
        """self.input_ids, self.n_word_nodes, self.n_entity_nodes, self.position_ids, self.attention_mask, self.masked_lm_labels, \
        self.ent_masked_lm_labels, self.rel_masked_lm_labels, self.token_type_ids = [], [], [], [], [], [], [], [], []"""
        self.path = path
        self.data = []
        self.__read_file__()
    
    def __read_file__(self):
        with open(self.path, 'r') as f:
            for line in f.readlines():
                conv = json.loads(line)
                for utt in conv['utterances']:
                    if len(utt['relations']) == 0:
                        continue
                    self.__add_element__(utt)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
 
    def __add_element_gt__(self,utt):
        token_types = []
        em_indices,personal_ids = utt['input_pec']['entity_mention_idx']
        ems = list(set(personal_ids))
        for em in ems:
            temp = utt['input_pec']['token_type_ids'].copy()
            for pers_id,em_index in zip(personal_ids,em_indices):
                if em == pers_id:
                    temp[em_index-1] = 5
            token_types.append(temp)
        return token_types

    def __add_element__(self,utt):
        types = self.__add_element_gt__(utt)
        for x in types:
            assert len(utt['input_pec']['token_type_ids']) == len(x)
            self.data.append({
                'input_ids': utt['input_pec']['nodes'],
                'attention_mask':utt['input_pec']['adj'],
                'token_type_ids':x,
                'position_ids':utt['input_pec']['soft_position'],
                'n_pkg_ents':utt['input_pec']['n_pkg_ents'],
                'n_pkg_rels':utt['input_pec']['n_pkg_rels'],
                'n_word_nodes':utt['input_pec']['n_word_nodes'],
                'n_relation_nodes':utt['input_pec']['n_relation_nodes'],
                'n_tail_mentions':utt['input_pec']['n_object_nodes'],
                'target':[0,1], #need actual fix
            })
            """self.input_ids.append(utt['input_pec']['nodes'])
            self.n_pkg_ents.append(utt['input_pec']['n_pkg_ents'])
            self.n_pkg_rels.append(utt['input_pec']['n_pkg_rels'])
            self.n_word_nodes.append(utt['input_pec']['n_word_nodes'])
            self.n_relation_nodes.append(utt['input_pec']['n_relation_nodes'])
            self.n_object_nodes.append(utt['input_pec']['n_object_nodes'])
            self.position_ids.append(utt['input_pec']['soft_position'])
            self.attention_mask.append(utt['input_pec']['adj'])
            assert len(utt['input_pec']['token_type_ids']) == len(x)
            self.token_type_ids.append(x)"""
    
    def collate_fn(self, batch):
        
        input_keys= ['input_ids','attention_mask','token_type_ids','position_ids','n_pkg_ents','n_pkg_rels','n_word_nodes','n_relation_nodes','n_tail_mentions']
        target_keys = ['target']
        max_pkg_ents, max_pkg_rels, max_words,max_rels,max_tails = 0,0,0,0,0
        batch_pkg_ents, batch_pkg_rels, batch_words, batch_rels, batch_object = [],[],[],[],[]
        
        batch_x = {n: [] for n in input_keys}
        batch_y = {n: [] for n in target_keys}
        
        for sample in batch:
            for n,v in sample.items():
                if n in input_keys:
                    batch_x[n].append(v)
                if n in target_keys:
                    batch_y[n].append(v)
            n_pkg_ents = sample['n_pkg_ents']
            n_pkg_rels = sample['n_pkg_rels']
            n_word_nodes = sample['n_word_nodes']
            n_relation_nodes = sample['n_relation_nodes']
            n_tail_mentions = sample['n_tail_mentions']
            pkg_ents = sample['input_ids'][0:n_pkg_ents]
            pkg_rels = sample['input_ids'][n_pkg_ents:n_pkg_ents+n_pkg_rels]
            word_nodes = sample['input_ids'][n_pkg_ents+n_pkg_rels:n_pkg_ents+n_pkg_rels+n_word_nodes]
            rel_nodes = sample['input_ids'][n_pkg_ents+n_pkg_rels+n_word_nodes:n_pkg_ents+n_pkg_rels+n_word_nodes+n_relation_nodes]
            tail_nodes = sample['input_ids'][n_pkg_ents+n_pkg_rels+n_word_nodes+n_relation_nodes:]
            
            
            """pkg_ents = [f(x) for x in pkg_ents]
            pkg_rels = [f(x) for x in pkg_rels]
            rel_nodes = [f(x) for x in rel_nodes]
            word_nodes = [f(x) for x in word_nodes]
            tail_nodes = [f(x) for x in tail_nodes]"""
            
            
            batch_pkg_ents.append(pkg_ents)
            batch_pkg_rels.append(pkg_rels)
            batch_words.append(word_nodes)
            batch_rels.append(rel_nodes)
            batch_object.append(tail_nodes)

            #batch_y['pkg_ent_seq_len'].append()
            max_pkg_ents = len(pkg_ents) if len(pkg_ents) > max_pkg_ents else max_pkg_ents
            max_pkg_rels = len(pkg_rels) if len(pkg_ents) > max_pkg_rels else max_pkg_rels
            max_words = len(word_nodes) if len(word_nodes) > max_words else max_words
            max_rels = len(rel_nodes) if len(rel_nodes) > max_rels else max_rels
            max_tails = len(tail_nodes) if len(tail_nodes) > max_tails else max_tails

        
        #Padding
        seq_len = max_pkg_ents + max_pkg_rels +max_words+max_rels+max_tails
        for i in range(len(batch_words)):
            pkg_ent_pad = max_pkg_ents - len(batch_pkg_ents[i])
            pkg_rels_pad = max_pkg_rels - len(batch_pkg_rels[i])
            word_pad = max_words - len(batch_words[i])
            rels_pad = max_rels - len(batch_rels[i])
            tail_pad = max_tails - len(batch_object[i])
            n_pkg_ents = batch_x['n_word_nodes'][i]
            n_pkg_rels = batch_x['n_pkg_rels'][i]
            n_word_nodes = batch_x['n_word_nodes'][i]
            n_relation_nodes = batch_x['n_relation_nodes'][i]
            n_tail_mentions = batch_x['n_tail_mentions'][i]
            batch_x['position_ids'][i] = batch_x['position_ids'][i][:n_pkg_ents] +[0]*pkg_ent_pad + \
                batch_x['position_ids'][i][n_pkg_ents:n_pkg_ents+n_pkg_rels] + [0]*pkg_rels_pad + \
                batch_x['position_ids'][i][n_pkg_ents+n_pkg_rels:n_pkg_ents+n_pkg_rels+n_word_nodes]+ [0]*word_pad +\
                batch_x['position_ids'][i][n_pkg_ents+n_pkg_rels+n_word_nodes:n_pkg_ents+n_pkg_rels+n_word_nodes+n_relation_nodes] + [0]*rels_pad + \
                batch_x['position_ids'][i][n_pkg_ents+n_pkg_rels+n_word_nodes+n_relation_nodes:] + [0] * tail_pad
            
            batch_x['token_type_ids'][i] = batch_x['token_type_ids'][i][:n_pkg_ents] +[0]*pkg_ent_pad + \
                batch_x['token_type_ids'][i][n_pkg_ents:n_pkg_ents+n_pkg_rels] + [0]*pkg_rels_pad + \
                batch_x['token_type_ids'][i][n_pkg_ents+n_pkg_rels:n_pkg_ents+n_pkg_rels+n_word_nodes]+ [0]*word_pad +\
                batch_x['token_type_ids'][i][n_pkg_ents+n_pkg_rels+n_word_nodes:n_pkg_ents+n_pkg_rels+n_word_nodes+n_relation_nodes] + [0]*rels_pad + \
                batch_x['token_type_ids'][i][n_pkg_ents+n_pkg_rels+n_word_nodes+n_relation_nodes:] + [0] * tail_pad
            
            adj = torch.tensor(batch_x['attention_mask'][i], dtype=torch.int)
            adj = torch.cat((adj[:n_pkg_ents, :],
                             torch.ones(pkg_ent_pad, adj.shape[1], dtype=torch.int),
                             adj[n_pkg_ents:n_pkg_ents+n_pkg_rels, :],
                             torch.ones(pkg_rels_pad, adj.shape[1], dtype=torch.int),
                             adj[n_pkg_ents+n_pkg_rels:n_pkg_ents+n_pkg_rels+n_word_nodes, :],
                             torch.ones(word_pad, adj.shape[1], dtype=torch.int),
                             adj[n_pkg_ents+n_pkg_rels+n_word_nodes:n_pkg_ents+n_pkg_rels+n_word_nodes+n_relation_nodes, :],
                             torch.ones(rels_pad, adj.shape[1], dtype=torch.int),
                             adj[n_pkg_ents+n_pkg_rels+n_word_nodes+n_relation_nodes:, :],
                             torch.ones(tail_pad, adj.shape[1], dtype=torch.int)),dim=0)
            assert adj.shape[0] == seq_len
            adj =torch.cat((adj[:,:n_pkg_ents],
                             torch.zeros(seq_len, pkg_ent_pad, dtype=torch.int),
                             adj[:,n_pkg_ents:n_pkg_ents+n_pkg_rels],
                             torch.zeros(seq_len,pkg_rels_pad, dtype=torch.int),
                             adj[:,n_pkg_ents+n_pkg_rels:n_pkg_ents+n_pkg_rels+n_word_nodes],
                             torch.zeros(seq_len,word_pad, dtype=torch.int),
                             adj[:,n_pkg_ents+n_pkg_rels+n_word_nodes:n_pkg_ents+n_pkg_rels+n_word_nodes+n_relation_nodes],
                             torch.zeros(seq_len,rels_pad, dtype=torch.int),
                             adj[:,n_pkg_ents+n_pkg_rels+n_word_nodes+n_relation_nodes:],
                             torch.zeros(seq_len,tail_pad, dtype=torch.int)),dim=1)
            batch_x['attention_mask'][i] = adj
        
        #Test code - NEEDS TO BE DELETED
        def f(x):
                if str(x).isnumeric():
                    return int(x)
                return 0
        
        for k,v in batch_x.items():
            if k == 'attention_mask':
                batch_x[k] = torch.stack(v, dim=0)
            elif k == 'input_ids':
                z = [f(x) for x in v]
                batch_x[k] = torch.tensor(z)
            else:
                batch_x[k] = torch.tensor(v)
        for k,v in batch_y.items():
            batch_y[k] =torch.tensor(v)
        return (batch_x, batch_y)

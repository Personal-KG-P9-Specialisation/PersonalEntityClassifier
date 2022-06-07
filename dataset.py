
from re import X
from torch.utils.data import Dataset
import json
import torch

class PKGDataSet(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = []
        self.__read_file__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    
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
        pers_ent, g_t = set(),[]
        pkg = utt['pkg']
        pkg = [tuple(x) for x in pkg]
        for sub,_,obj in pkg:
            for i in [sub,obj]:
                if not str(i).startswith('c_'):
                    pers_ent.add(i)
        token_types = []
        em_indices,personal_ids = utt['input_pec']['entity_mention_idx']
        ems = list(set(personal_ids))
        for em in ems:
            temp = utt['input_pec']['token_type_ids'].copy()
            for pers_id,em_index in zip(personal_ids,em_indices):
                if em == pers_id:
                    temp[em_index-1] = 5
            token_types.append(temp)

            if em in pers_ent:
                g_t.append([1,0])
            else:
                g_t.append([0,1])
        return token_types, g_t

    def __add_element__(self,utt):
        types, gt = self.__add_element_gt__(utt)
        for x,y in zip(types,gt):
            assert len(utt['input_pec']['token_type_ids']) == len(x)
            
            self.data.append({
                'input_ids': utt['input_pec']['nodes'],
                'attention_mask':utt['input_pec']['adj'],
                'token_type_ids':x,
                'position_ids':utt['input_pec']['soft_position'],
                'n_pkg_ents':utt['input_pec']['n_pkg_ents'],
                'n_pkg_cskg':utt['input_pec']['n_pkg_cskg'],
                'n_pkg_rels':utt['input_pec']['n_pkg_rels'],
                'n_word_nodes':utt['input_pec']['n_word_nodes'],
                'n_relation_nodes':utt['input_pec']['n_relation_nodes'],
                'n_tail_mentions':utt['input_pec']['n_object_nodes'],
                'target':y,
            })
    
    def collate_fn(self, batch):
        
        input_keys= ['target','input_ids','attention_mask','token_type_ids','position_ids','n_pkg_ents','n_pkg_cskg','n_pkg_rels','n_word_nodes','n_relation_nodes','n_tail_mentions']
        target_keys = ['target']
        max_pkg_ents,max_pkg_cskg, max_pkg_rels, max_words,max_rels,max_tails = 0,0,0,0,0,0
        batch_pkg_ents, batch_pkg_rels, batch_words, batch_rels, batch_object,batch_start_tokens,batch_pkg_cskg_ents,batch_sep_token = [],[],[],[],[],[],[],[]
        
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
            n_pkg_cskg = sample['n_pkg_cskg']
            n_word_nodes = sample['n_word_nodes']
            n_relation_nodes = sample['n_relation_nodes']
            n_tail_mentions = sample['n_tail_mentions']
            start_tokens = sample['input_ids'][0]
            pkg_ents = sample['input_ids'][1:n_pkg_ents]
            pkg_cskg_ents = sample['input_ids'][n_pkg_ents+1:n_pkg_ents+1+n_pkg_cskg]
            pkg_rels = sample['input_ids'][n_pkg_ents+1+n_pkg_cskg:n_pkg_ents+1+n_pkg_cskg+n_pkg_rels]
            sep_token = sample['input_ids'][n_pkg_ents+1+n_pkg_cskg+n_pkg_rels:n_pkg_ents+2+n_pkg_cskg+n_pkg_rels]
            word_nodes = sample['input_ids'][n_pkg_ents+2+n_pkg_cskg+n_pkg_rels:n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes]
            rel_nodes = sample['input_ids'][n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes:n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes+n_relation_nodes]
            tail_nodes = sample['input_ids'][n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes+n_relation_nodes:]
            
            #Test code - NEEDS TO BE DELETED
            """def f(x):
                if str(x).isnumeric():
                    return int(x)
                return 0
            pkg_ents = [f(x) for x in pkg_ents]
            pkg_rels = [f(x) for x in pkg_rels]
            rel_nodes = [f(x) for x in rel_nodes]
            word_nodes = [f(x) for x in word_nodes]
            tail_nodes = [f(x) for x in tail_nodes]"""
            
            batch_start_tokens.append(start_tokens)
            batch_pkg_ents.append(pkg_ents)
            batch_pkg_cskg_ents.append(pkg_cskg_ents)
            batch_pkg_rels.append(pkg_rels)
            batch_sep_token.append(sep_token)
            batch_words.append(word_nodes)
            batch_rels.append(rel_nodes)
            batch_object.append(tail_nodes)

            #batch_y['pkg_ent_seq_len'].append()
            max_pkg_ents = len(pkg_ents) if len(pkg_ents) > max_pkg_ents else max_pkg_ents
            max_pkg_rels = len(pkg_rels) if len(pkg_rels) > max_pkg_rels else max_pkg_rels
            max_pkg_cskg =len(pkg_cskg_ents) if len(pkg_cskg_ents) > max_pkg_cskg else max_pkg_cskg
            max_words = len(word_nodes) if len(word_nodes) > max_words else max_words
            max_rels = len(rel_nodes) if len(rel_nodes) > max_rels else max_rels
            max_tails = len(tail_nodes) if len(tail_nodes) > max_tails else max_tails

        
        #Padding
        seq_len = max_pkg_ents +max_pkg_cskg+ max_pkg_rels +max_words+max_rels+max_tails+2
        for i in range(len(batch_words)):
            pkg_ent_pad = max_pkg_ents - len(batch_pkg_ents[i])
            pkg_cskg_pad = max_pkg_cskg - len(batch_pkg_cskg_ents[i])
            pkg_rels_pad = max_pkg_rels - len(batch_pkg_rels[i])
            word_pad = max_words - len(batch_words[i])
            rels_pad = max_rels - len(batch_rels[i])
            tail_pad = max_tails - len(batch_object[i])

            pkg_ent_vec =batch_pkg_ents[i] + [1] * pkg_ent_pad
            pkg_cskg_vec = batch_pkg_cskg_ents[i] + [1] * pkg_cskg_pad
            pkg_rel_vec = batch_pkg_rels[i] + [1] * pkg_rels_pad
            word_vec = batch_words[i] + [1] * word_pad
            rel_batch_vec =batch_rels[i] + [1] * rels_pad
            object_batch_vec = batch_object[i] + [1] * tail_pad
            batch_x['input_ids'][i] = [batch_start_tokens[i]]+pkg_ent_vec + \
                                      pkg_cskg_vec + \
                                      pkg_rel_vec + \
                                      [2] + word_vec + \
                                      rel_batch_vec +  \
                                      object_batch_vec #batch_object[i] + [1] * tail_pad
            
            
            
            n_pkg_ents = batch_x['n_pkg_ents'][i]
            n_pkg_cskg = batch_x['n_pkg_cskg'][i]
            n_pkg_rels = batch_x['n_pkg_rels'][i]
            n_word_nodes = batch_x['n_word_nodes'][i]
            n_relation_nodes = batch_x['n_relation_nodes'][i]
            n_tail_mentions = batch_x['n_tail_mentions'][i]
            batch_x['position_ids'][i] = [batch_x['position_ids'][i][0]]+ batch_x['position_ids'][i][1:n_pkg_ents] +[0]*pkg_ent_pad + \
                batch_x['position_ids'][i][n_pkg_ents+1:n_pkg_ents+1+n_pkg_cskg] + [0]*pkg_cskg_pad + \
                batch_x['position_ids'][i][n_pkg_ents+1+n_pkg_cskg:n_pkg_ents+1+n_pkg_cskg+n_pkg_rels] + [0]*pkg_rels_pad + \
                batch_x['position_ids'][i][n_pkg_ents+1+n_pkg_cskg+n_pkg_rels:n_pkg_ents+2+n_pkg_cskg+n_pkg_rels] +   \
                batch_x['position_ids'][i][n_pkg_ents+2+n_pkg_cskg+n_pkg_rels:n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes]+ [0]*word_pad +\
                batch_x['position_ids'][i][n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes:n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes+n_relation_nodes] + [0]*rels_pad + \
                batch_x['position_ids'][i][n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes+n_relation_nodes:] + [0] * tail_pad
            batch_x['token_type_ids'][i] = [batch_x['token_type_ids'][i][0]]+ batch_x['token_type_ids'][i][1:n_pkg_ents] +[0]*pkg_ent_pad + \
                batch_x['token_type_ids'][i][n_pkg_ents+1:n_pkg_ents+1+n_pkg_cskg] + [0]*pkg_cskg_pad + \
                batch_x['token_type_ids'][i][n_pkg_ents+1+n_pkg_cskg:n_pkg_ents+1+n_pkg_cskg+n_pkg_rels] + [0]*pkg_rels_pad + \
                batch_x['token_type_ids'][i][n_pkg_ents+1+n_pkg_cskg+n_pkg_rels:n_pkg_ents+2+n_pkg_cskg+n_pkg_rels] +   \
                batch_x['token_type_ids'][i][n_pkg_ents+2+n_pkg_cskg+n_pkg_rels:n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes]+ [0]*word_pad +\
                batch_x['token_type_ids'][i][n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes:n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes+n_relation_nodes] + [0]*rels_pad + \
                batch_x['token_type_ids'][i][n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes+n_relation_nodes:] + [0] * tail_pad
            """batch_x['token_type_ids'][i] = batch_x['token_type_ids'][i][:n_pkg_ents] +[0]*pkg_ent_pad + \
                batch_x['token_type_ids'][i][n_pkg_ents:n_pkg_ents+n_pkg_rels] + [0]*pkg_rels_pad + \
                batch_x['token_type_ids'][i][n_pkg_ents+n_pkg_rels:n_pkg_ents+n_pkg_rels+n_word_nodes]+ [0]*word_pad +\
                batch_x['token_type_ids'][i][n_pkg_ents+n_pkg_rels+n_word_nodes:n_pkg_ents+n_pkg_rels+n_word_nodes+n_relation_nodes] + [0]*rels_pad + \
                batch_x['token_type_ids'][i][n_pkg_ents+n_pkg_rels+n_word_nodes+n_relation_nodes:] + [0] * tail_pad"""
            
            adj = torch.tensor(batch_x['attention_mask'][i], dtype=torch.int)
            #print(torch.reshape(adj[0,:],(1,-1)).shape, adj[1:n_pkg_ents, :].shape)
            adj = torch.cat((torch.reshape(adj[0,:],(1,-1)),adj[1:n_pkg_ents, :],
                             torch.ones(pkg_ent_pad, adj.shape[1], dtype=torch.int),
                             adj[n_pkg_ents+1:n_pkg_ents+1+n_pkg_cskg, :],
                             torch.ones(pkg_cskg_pad, adj.shape[1], dtype=torch.int),
                             adj[n_pkg_ents+1+n_pkg_cskg:n_pkg_ents+1+n_pkg_cskg+n_pkg_rels, :],
                             torch.ones(pkg_rels_pad, adj.shape[1], dtype=torch.int),
                             adj[n_pkg_ents+1+n_pkg_cskg+n_pkg_rels:n_pkg_ents+2+n_pkg_cskg+n_pkg_rels,:],
                             adj[n_pkg_ents+2+n_pkg_cskg+n_pkg_rels:n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes, :],
                             torch.ones(word_pad, adj.shape[1], dtype=torch.int),
                             adj[n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes:n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes+n_relation_nodes, :],
                             torch.ones(rels_pad, adj.shape[1], dtype=torch.int),
                             adj[n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes+n_relation_nodes:, :],
                             torch.ones(tail_pad, adj.shape[1], dtype=torch.int)),dim=0)
            assert adj.shape[0] == seq_len
            adj =torch.cat((torch.reshape( adj[:,0],(-1,1)),adj[:,1:n_pkg_ents],
                             torch.zeros(seq_len, pkg_ent_pad, dtype=torch.int),
                             adj[:,n_pkg_ents+1:n_pkg_ents+1+n_pkg_cskg],
                             torch.zeros(seq_len,pkg_cskg_pad, dtype=torch.int),
                             adj[:,n_pkg_ents+1+n_pkg_cskg:n_pkg_ents+1+n_pkg_cskg+n_pkg_rels],
                             torch.zeros(seq_len,pkg_rels_pad, dtype=torch.int),
                             adj[:,n_pkg_ents+1+n_pkg_cskg+n_pkg_rels:n_pkg_ents+2+n_pkg_cskg+n_pkg_rels],
                             adj[:,n_pkg_ents+2+n_pkg_cskg+n_pkg_rels:n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes],
                             torch.zeros(seq_len,word_pad, dtype=torch.int),
                             adj[:,n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes:n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes+n_relation_nodes],
                             torch.zeros(seq_len,rels_pad, dtype=torch.int),
                             adj[:,n_pkg_ents+2+n_pkg_cskg+n_pkg_rels+n_word_nodes+n_relation_nodes:],
                             torch.zeros(seq_len,tail_pad, dtype=torch.int)),dim=1)
            batch_x['attention_mask'][i] = adj
            
            batch_x['n_pkg_ents'][i] = max_pkg_ents
            batch_x['n_pkg_cskg'][i] = max_pkg_cskg
            batch_x['n_pkg_rels'][i] = max_pkg_rels
            batch_x['n_word_nodes'][i] = max_words
            batch_x['n_relation_nodes'][i] = max_rels
            batch_x['n_tail_mentions'][i] = max_tails
        
        
        for k,v in batch_x.items():
            if k == 'attention_mask':
                batch_x[k] = torch.stack(v, dim=0)
            elif k == 'target':
                batch_x[k] =torch.tensor(v, dtype=torch.float32)
            else:
                batch_x[k] = torch.tensor(v)
            batch_x[k] = batch_x[k].to(self.device)
        for k,v in batch_y.items():
            batch_y[k] =torch.tensor(v, dtype=torch.float32)
            #batch_y[k] = batch_y[k].to(self.device)
        return (batch_x, batch_y)

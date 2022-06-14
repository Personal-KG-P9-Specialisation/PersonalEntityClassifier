from utils import create_input
import json
from transformers import RobertaTokenizer
import torch
from dataset import UtterancePadder


def add_element_gt__(utt, em):
        pers_ent, g_t = set(),[]
        pkg = utt['pkg']
        pkg = [tuple(x) for x in pkg]
        for sub,_,obj in pkg:
            for i in [sub,obj]:
                if not str(i).startswith('c_'):
                    pers_ent.add(i)
        token_types = []
        em_indices,personal_ids = utt['input_pec']['entity_mention_idx']
        temp = utt['input_pec']['token_type_ids'].copy()
        for pers_id,em_index in zip(personal_ids,em_indices):
                if em == pers_id:
                    temp[em_index-1] = 5
        token_types.append(temp)

        if em in pers_ent:
                g_t.append(1)
        else:
                g_t.append(0)
        return token_types, g_t
def convert_utt_to_dict(utt, em):
    types,gt = add_element_gt__(utt,em)
    return {
                'input_ids': utt['input_pec']['nodes'],
                'attention_mask':utt['input_pec']['adj'],
                'token_type_ids':types[0],
                'position_ids':utt['input_pec']['soft_position'],
                'n_pkg_ents':utt['input_pec']['n_pkg_ents'],
                'n_pkg_cskg':utt['input_pec']['n_pkg_cskg'],
                'n_pkg_rels':utt['input_pec']['n_pkg_rels'],
                'n_word_nodes':utt['input_pec']['n_word_nodes'],
                'n_relation_nodes':utt['input_pec']['n_relation_nodes'],
                'n_tail_mentions':utt['input_pec']['n_object_nodes'],
                'target':gt,
    }

def evaluate_convs(data_path,model_path):
    convs = []
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path)
    model = model.to(device)
    u = UtterancePadder(24,42,16,24,5,19)
    with open(data_path,'r') as f:
        for line in f.readlines():
            convs.append(json.loads(line))
    for conv in convs:
        for utt in conv['utterances']:
            #print(utt)
            #input = create_input(tokenizer, utt, 10000)
            
            """print(input['entity_mention_idx'])
            print(input.keys())
            nodes = input['nodes']
            print(len(nodes))
            em_nodes = [nodes[x] for x in input['entity_mention_idx'][0] if len(nodes)>x]
            print(tokenizer.decode( em_nodes))
            print(input.keys())"""
            #1 is personal id for first utterance.
            input2= convert_utt_to_dict(utt,1)
            #print(input2.keys())
            
            i = u.pad_utt([input2])
            
            p = model(i['input_ids'],i['attention_mask'],i['token_type_ids'],i['position_ids'],None, None,i['n_pkg_ents'],i['n_pkg_cskg'],i['n_pkg_rels'],i['n_word_nodes'],i['n_relation_nodes'],i['n_tail_mentions'],i['target'])
            print(p)
            print(i)
            print(f"Prediction is {p['pred'][0]}, while ground truth is {i['target'][0]}")
            exit()
            print()
evaluate_convs('data/input3.jsonl','models3_w_seed/best_URG_Sig_f_score_2022-06-12-10-50-09-248443')

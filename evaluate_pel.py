from codecs import lookup_error
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

def prepare_data(data_paths,model_path, output_path):
    convs = []
    rights = 0
    all = 0
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path)
    model = model.to(device)
    u = UtterancePadder(24,42,16,24,5,19)
    for i in data_paths:
        with open(i,'r') as f:
            for line in f.readlines():
                convs.append(json.loads(line))
    f = open(output_path,'w')
    for conv in convs:
        conv['utterances'] = sorted(conv['utterances'], key =lambda x: x['turn'])
        textsofar = ''
        for utt in conv['utterances']:
            utt['dialogue_history'] = textsofar
            textsofar += utt['text']
            personal_ids = []
            for rel in utt['relations']:
                for em in [rel['head_span'],rel['child_span']]:
                    if not em['text'].lower() == ['i', 'my','he','his','her','they','their','our','we', 'she','hers']:
                        personal_ids.append(em['personal_id'])
            lookup_pers = {}
            for x in personal_ids:
                input2= convert_utt_to_dict(utt,x)
                
                i = u.pad_utt([input2])
                p = model(i['input_ids'],i['attention_mask'],i['token_type_ids'],i['position_ids'],None, None,i['n_pkg_ents'],i['n_pkg_cskg'],i['n_pkg_rels'],i['n_word_nodes'],i['n_relation_nodes'],i['n_tail_mentions'],i['target'])
                
                lookup_pers[x] = (p['pred'].tolist()[0],i['target'].tolist()[0])
                if p['pred'][0] > 0.5:
                    if int(i['target'][0]) == 1:
                            rights += 1
                else:
                    if int(i['target'][0]) == 0:
                        rights += 1
                all += 1
            for rel in utt['relations']:
                for em in [rel['head_span'],rel['child_span']]:
                    if not em['text'].lower() == ['i', 'my','he','his','her','they','their','our','we', 'she','hers']:
                        if em['personal_id'] in lookup_pers.keys():
                            em['pec_prediction']= lookup_pers[em['personal_id']]
        f.write(json.dumps(conv)+'\n')
    f.close()
    print(f"Accuracy on this data is {rights/all}%") 

if __name__ == "__main__":
    prepare_data(['data/input2.jsonl','data/input3.jsonl'],'final_model/best_URG_Sig_f_score_2022-06-15-05-43-50-890178', 'pel_test_data.jsonl')

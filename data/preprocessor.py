import json
#docker run -it --name dev_pers -v "$(pwd)"/:/code pers_cls:1 bin/bash
def remove_duplicate_from_lst_dict(l):
    seen = set()
    new_l = []
    for d in l:
        t = tuple(d.items())
        if t not in seen:
            seen.add(t)
            new_l.append(d)
    return new_l
def spanEquals(span1,span2):
    if span1['start'] == span2['start'] and span1['end'] == span2['end']:
        return True
    return False

#TODO: map spans from utterances to relations span instead.
#TODO: reidentify personal entities.
#TODO: readjust span start and end indices.
class Preprocessor:
    def __init__(self, triple_file, CSKG_file, personal_file):
        self.triple_file = triple_file
        self.CSKG_file = CSKG_file
        self.personal_file = personal_file
        self.convs = []
        with open(triple_file, "r") as f:
            for line in f.readlines():
                self.convs.append(json.loads(line))

    def filter_spans(self):
        for conv in self.convs:
            for utt in conv['utterances']:
                utt['spans'] = remove_duplicate_from_lst_dict(utt['spans'])
        
    def add_conceptnet(self):
        f = open(self.CSKG_file, 'r')
        convs = []
        for conv in self.convs:
            new_conv = self.__add_single_conceptnet_to_conv( conv,f)
            f.seek(0)
            convs.append(new_conv)
        self.convs = convs
        #print(convs)
        #f.seek(0) for resetting pointer head
        f.close()
    
    def __add_single_conceptnet_to_conv(self,conv, file_pointer):
        for line in file_pointer.readlines():
            temp = json.loads(line)
            if temp['conv_id'] != conv['conv_id']:
                continue
            for utt in conv['utterances']:
                if utt['turn'] == temp['turn']:
                    em = temp['spans'][0]
                    t = True
                    for x in utt['spans']:
                        if x == em:
                            try:
                                if temp['accept'][0] not in ['NIL_otherLink','NIL_ambiguous']:
                                    x['conceptnet'] = temp['accept'][0]
                            except IndexError:
                                pass
                            t = False
                    if t:
                        raise("something went wrong ", em)
        return conv
    
    def add_personal_entities(self):
        f = open(self.personal_file, 'r')
        convs = []
        for conv in self.convs:
            convs.append(self.__add_single_pers_ent(conv,f))
            f.seek(0)
        f.close()
        
        self.convs = self.__add_missing_for_pers_id( convs)
    def __add_missing_for_pers_id(self, convs):
        idx = 1
        pers_idx = []
        for conv in convs:
            for utt in conv['utterances']:
                for span in utt['spans']:
                    if 'personal_id' in span.keys():
                        pers_idx.append(span['personal_id'])
        for conv in convs:
            for utt in conv['utterances']:
                for span in utt['spans']:
                    if not 'personal_id'in span.keys():
                        while(idx in pers_idx):
                            idx += 1
                        span['personal_id'] = idx
                        pers_idx.append(idx)
        return convs
    
    def __add_single_pers_ent(self, conv, file_pointer):
        def assign_personal_id(adj_matrix, span_map, span, idx):
            if not 'personal_id' in span.keys():
                span['personal_id'] = idx
                if (span['start'],span['end']) in adj_matrix.keys():
                    for adj_tuple in adj_matrix[(span['start'],span['end'])]:
                        adj_span = span_map[adj_tuple]
                        assign_personal_id(adj_matrix, span_map, adj_span, idx)
        data = None
        for line in file_pointer.readlines():
            temp = json.loads(line)
            if temp['conv_id'] == conv['conv_id']:
                data = temp
        adj_matrix = {}
        span_map = {}
        
        for rel in data['relations']:
            head = rel['head_span']
            head['text'] = data['text'][head['start']:head['end']]
            child = rel['child_span']
            child['text'] = data['text'][child['start']:child['end']]
            
            if (head['start'],head['end']) in adj_matrix.keys():
                adj_matrix[(head['start'],head['end'])].append((child['start'],child['end']))
            else:
                adj_matrix[(head['start'],head['end'])] = [(child['start'],child['end'])]
            span_map[(head['start'],head['end'])] = head
            span_map[(child['start'],child['end'])] = child
        
        counter = 0
        for rel in data['relations']:
            counter += 1
            head = rel['head_span']
            child = rel['child_span']
            assign_personal_id(adj_matrix, span_map, head, counter)
            assign_personal_id(adj_matrix, span_map, child, counter)
        for rel in data['relations']:
            head = rel['head_span']
            child = rel['child_span']
        for utt in conv['utterances']:
            utt_start_index = data['text'].find(utt['text'])
            utt_fin_index = utt_start_index + len(utt['text'])
            
            for key in span_map.keys():
                span_map[key]
                if utt_start_index <= key[0] and key[1] <= utt_fin_index:
                    for span in utt['spans']:
                        if span['text'] == data['text'][key[0]:key[1]]:
                            span['personal_id'] = span_map[key]['personal_id']
        return conv
    
    def divide_convs_two_ways(self):
        convs = []
        for conv in self.convs:
            conv1 = {'conv_id':conv['conv_id'], 'utterances':[], 'Agent': 1}
            conv2 = {'conv_id':conv['conv_id'], 'utterances':[], 'Agent': 2}
            even = False
            for utt in conv['utterances']:
                if even:
                    conv2['utterances'].append(utt)
                    even=False
                else:
                    even=True
                    conv1['utterances'].append(utt)
            convs.append(conv1)
            convs.append(conv2)
        self.convs = convs
    def export_convs(self,file):
        with open(file,'w') as f:
            for conv in self.convs:
                f.write(json.dumps(conv)+'\n')
        print("Conversations Exported!!!")

def map_span_to_relations(file_path, output_file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    for conv in data:
        for utt in conv['utterances']:
            for rel in utt['relations']:
                for span in utt['spans']:
                    if rel['head_span']['start'] == span['start'] and rel['head_span']['end'] == span['end']:
                        rel['head_span'] = span
                    if rel['child_span']['start'] == span['start'] and rel['child_span']['end'] == span['end']:
                        rel['child_span'] = span
    with open(output_file_path, "w") as f:
        for conv in data:
            f.write(json.dumps(conv)+'\n')
    

if __name__ =="__main__":
    p = Preprocessor('/data/final_updated_filtered_relation_annotated_triples.jsonl','/data/final_annotated_conceptnet_entities.jsonl','/data/final_annotated_personal_entities.jsonl')
    p.filter_spans()
    p.add_conceptnet()
    p.divide_convs_two_ways()
    p.add_personal_entities()
    p.export_convs('total_dataset2.jsonl')
    map_span_to_relations("total_dataset2.jsonl",'total_dataset2.jsonl')

"""
format of triples:
[
    conv_id: ..
    utterances : [
        {
            turn :0
            conv_id:0
            text: ''
            relations: triples
        }
    ]
]
format of CSKG:
[
    text: utterance text
    conv_id : 
    turn: 
    spans:[] #entity mention with one element
    accept: ConceptNet entity
]
"""
import json
import matplotlib.pyplot as plt

def num_conv(data):
    return len(data)

def num_utterance(data):
    total = 0
    for conv in data:
        total = total + len(conv['utterances'])
    return total

def num_rel(data):
    total = 0
    for conv in data:
        for utt in conv['utterances']:
            total = total + len(utt['relations'])
    return total

def num_cskg(data):
    total = 0
    for conv in data:
        for utt in conv['utterances']:
            for rel in utt['relations']:
                for i in [rel['head_span'], rel['child_span']]:
                    if 'conceptnet' in i.keys():    
                        total = total + 1
    return total

def num_cskg_unique(data):
    unique_list = []
    for conv in data:
        for utt in conv['utterances']:
            for rel in utt['relations']:
                for i in [rel['head_span'], rel['child_span']]:
                    if 'conceptnet' in i.keys():    
                        if i['conceptnet'] not in unique_list:
                            unique_list.append(i['conceptnet'])
    return len(unique_list)

def num_personal_ents(data):
    count = 0
    for conv in data:
        personal_ids = []
        for utt in conv['utterances']:
            for rel in utt['relations']:
                for i in [rel['head_span'], rel['child_span']]:
                    if 'personal_id' in i.keys():    
                        if i['personal_id'] not in personal_ids:
                            personal_ids.append(i['personal_id'])
        count += len(personal_ids)
    return count
def pers_chain(data):
    chains_freq = {}
    for conv in data:
        count = {}
        for utt in conv['utterances']:
            for rel in utt['relations']:
                for i in [rel['head_span'], rel['child_span']]:
                    # if utt['text'][i['start']:i['end']].lower() not in ["my", "i", "our", "we"]:
                    if 'personal_id' in i.keys():
                        if i['personal_id'] not in count.keys():
                            count[i['personal_id']] = 1
                        else:
                            count[i['personal_id']] = count[i['personal_id']] + 1 
        for val in count.values():
            if val not in chains_freq.keys():
                chains_freq[val] = 1
            else:
                chains_freq[val] = chains_freq[val] + 1
    return chains_freq

def histogram(dic):
    #dic.pop(1)
    #dic.pop(2)
    plt.bar(dic.keys(), dic.values(), color=(0.2, 0.4, 0.6, 0.6))
    plt.xlabel("Number of reoccuring personal entities")
    plt.ylabel("Frequency")
    plt.savefig('chains-freq1.png')

if __name__ == '__main__':
    with open('data/pec_convs.jsonl') as f:
        data = [json.loads(line) for line in f]
        #print(f"Number of conversations: {num_conv(data)}\n")
        #print(f"Number of Utterances: {num_utterance(data)}\n")
        #print(f"Number of triples: {num_rel(data)}\n")
        #print(f"Number of unique CSKG: {num_cskg_unique(data)}\n")
        #print(f"Number of unique personal entities: {num_personal_ents(data)}\n")
        histogram(pers_chain(data))
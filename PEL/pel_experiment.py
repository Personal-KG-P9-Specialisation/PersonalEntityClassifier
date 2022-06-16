import difflib
from importlib_metadata import Lookup
from stanfordcorenlp import StanfordCoreNLP
import spacy
import neuralcoref
import json
from sklearn.metrics import f1_score,precision_score, recall_score

def neural_coref(text, nlp):
    coref_chains = nlp(text)
    return coref_chains
def stanford_coref(text):
    # RUN THIS FIRST FROM NLP FOLDER: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 15000 
    nlp = StanfordCoreNLP('http://localhost', port=9001)
    coref_chains = nlp.coref(text)
    #print(coref_chains)
    nlp.close()
    chains = []
    for x in coref_chains:
        chains.append ([z[3] for z in x])
    return chains

def PEL_for_experiment(lookup, dialogue, utterance, mention, classification):   
    nlp = spacy.load('en')
    neuralcoref.add_to_pipe(nlp)
    if classification == "pronoun":
        personalPronouns = ["my", "i", "our", "we", "us"]
        alternativePronouns = ["he", "she", "you", "it", "its", "they", "them", "their", "theirs", "his", "hers", "her", "him", "your"]

        if mention['text'].lower() in personalPronouns:
            return mention['personal_id'],lookup
        elif mention['text'].lower() in alternativePronouns:
            dialogue_hist =dialogue + utterance['text']
            chains = neural_coref(dialogue_hist,nlp)
            for chain in chains._.coref_clusters:
                for element in chain.mentions:
                    if mention['text'] in dialogue_hist[element.start:element.end]: #element.start == (len(dialogue)+mention['start']) and element.end == (len(dialogue)+mention['end']):
                        entityMentions = [x for x in chain.mentions if x.lower() not in alternativePronouns]
                        match = difflib.get_close_matches(entityMentions[-1][-1], lookup.keys(), n=1)
                        if match[0].lower() in lookup.keys():
                            return lookup[match[0].lower()], lookup
            return None,lookup

    elif classification == "new":
        #for new entities we just use the existing id to check for.
        lookup[mention['text'].lower()] = mention['personal_id']
        return mention['personal_id'], lookup

    elif classification == "existing":
        match = difflib.get_close_matches(mention['text'], lookup.keys(), n=1)
        if match:
            if match[0].lower() is not mention['text']:
                lookup[mention['text'].lower()] = lookup[match[0].lower()]
                return lookup[match[0].lower()],lookup
            elif match[0].lower() in lookup.keys():
                return lookup[match[0].lower()], lookup
        # Next 3 lines might be removed - edge case: "existing" classification though not existing.
        idx = max(lookup.values()) + 1
        lookup[mention['text'].lower()] = idx
        return idx,lookup

def PEL_for_experiment_stan(lookup, dialogue, utterance, mention, classification):
    if classification == "pronoun":
        personalPronouns = ["my", "i", "our", "we", "us"]
        alternativePronouns = ["he", "she", "you", "it", "its", "they", "them", "their", "theirs", "his", "hers", "her", "him", "your"]

        if mention['text'].lower() in personalPronouns:
            return mention['personal_id'],lookup
        elif mention['text'].lower() in alternativePronouns:
            dialogue_hist =dialogue + utterance['text']
            chains = stanford_coref(dialogue_hist)
            for chain in chains:
                for element in chain:
                    if mention['text'] in element or mention['text'] == element: #element.start == (len(dialogue)+mention['start']) and element.end == (len(dialogue)+mention['end']):
                        entityMentions = [x for x in chain if x.lower() not in alternativePronouns]
                        match = difflib.get_close_matches(entityMentions[-1][-1], lookup.keys(), n=1)
                        if match == []:
                            return None, lookup
                        if match[0].lower() in lookup.keys():
                            return lookup[match[0].lower()], lookup
            # Next 3 lines might be removed - edge case: "existing" classification though not existing.
            #idx = max(lookup.values()) + 1
            #lookup[mention.lower()] = idx
            return None,lookup

    elif classification == "new":
        #for new entities we just use the existing id to check for.
        lookup[mention['text'].lower()] = mention['personal_id']
        return mention['personal_id'], lookup

    elif classification == "existing":
        match = difflib.get_close_matches(mention['text'], lookup.keys(), n=1)
        if match:
            if match[0].lower() is not mention['text']:
                lookup[mention['text'].lower()] = lookup[match[0].lower()]
                return lookup[match[0].lower()],lookup
            elif match[0].lower() in lookup.keys():
                return lookup[match[0].lower()], lookup
        # Next 3 lines might be removed - edge case: "existing" classification though not existing.
        idx = max(lookup.values()) + 1
        lookup[mention['text'].lower()] = idx
        return idx,lookup


def experiment(data_path):
    convs = []
    nones = 0
    with open(data_path,'r') as f:
        for line in f.readlines():
            convs.append(json.loads(line))
    prediction, ground_truth = [],[]
    for conv_nr,conv in enumerate(convs):
        lookup = { "i":1 }
        conv['utterances'] = sorted(conv['utterances'], key =lambda x: x['turn'])
        corefs = 0
        #we skip the ones with intial pkg for now. These require special attention to lookup table
        #if not conv['utterances'][0]['pkg'] == []:
        #    continue
        for utt in conv['utterances']:
            for rel in utt['relations']:
                if rel == []:
                    continue
                for em in [rel['head_span'], rel['child_span']]:
                    #we do not test user node
                    #if em['text'].lower() in ["my", "i", "our", "we", "us"]:
                    #    continue

                    if 'isPronoun' in em.keys():
                        pid,lookup = PEL_for_experiment(lookup, utt['dialogue_history'], utt, em, 'pronoun')
                        corefs +=1
                        if not pid is None:
                            prediction.append(int(pid))
                            ground_truth.append(int(em['personal_id']))
                        else:
                            nones += 1
                    elif em['pec_prediction'][0] > 0.5:
                        pid,lookup =PEL_for_experiment(lookup, utt['dialogue_history'], utt, em, "existing")
                        prediction.append(int(pid))
                        ground_truth.append(int(em['personal_id']))
                    else:
                        pid,lookup = PEL_for_experiment(lookup, utt['dialogue_history'], utt, em, "new")
                        prediction.append(int(pid))
                        ground_truth.append(int(em['personal_id']))
                        f1_score,precision_score, recall_score
        print(f"For conv {conv_nr} of {len(convs)}: Precision {precision_score(ground_truth,prediction, average='macro')}, Recall {recall_score(ground_truth,prediction, average='macro')}, F1-scor {f1_score(ground_truth,prediction, average='macro')}\n")
    print(f"There are {nones} None, and {corefs} Coref")

def pkg(data_path):
    convs = []
    nones = 0
    with open(data_path,'r') as f:
        for line in f.readlines():
            convs.append(json.loads(line))
    prediction, ground_truth = [],[]
    for conv_nr,conv in enumerate(convs):
        lookup = { "i":1 }
        conv['utterances'] = sorted(conv['utterances'], key =lambda x: x['turn'])
        corefs = 0
        #we skip the ones with intial pkg for now. These require special attention to lookup table
        #if not conv['utterances'][0]['pkg'] == []:
        #    continue
        pkg = set()
        for utt in conv['utterances']:
            for rel in utt['relations']:
                if rel == []:
                    continue
                for em in [rel['head_span'], rel['child_span']]:
                    #we do not test user node
                    #if em['text'].lower() in ["my", "i", "our", "we", "us"]:
                    #    continue
                    predicted_p_id = None
                    if 'isPronoun' in em.keys():
                        pid,lookup = PEL_for_experiment(lookup, utt['dialogue_history'], utt, em, 'pronoun')
                        predicted_p_id = pid
                        corefs +=1
                        if not pid is None:
                            prediction.append(int(pid))
                            ground_truth.append(int(em['personal_id']))
                        else:
                            nones += 1
                    elif em['pec_prediction'][0] > 0.5:
                        pid,lookup =PEL_for_experiment(lookup, utt['dialogue_history'], utt, em, "existing")
                        predicted_p_id=pid
                        prediction.append(int(pid))
                        ground_truth.append(int(em['personal_id']))
                    else:
                        pid,lookup = PEL_for_experiment(lookup, utt['dialogue_history'], utt, em, "new")
                        predicted_p_id = pid
                        prediction.append(int(pid))
                        ground_truth.append(int(em['personal_id']))
                    em['predicted_personal_id'] =predicted_p_id
                    predicted_p_id = None
                pkg.add(((rel['head_span']['predicted_personal_id'],rel['head_span']['personal_id']),rel['label'],(rel['child_span']['predicted_personal_id'],rel['child_span']['personal_id'])))
                for em in [rel['head_span'], rel['child_span']]:
                    if 'conceptnet' in em.keys():
                        pkg.add(((em['predicted_personal_id'],em['personal_id']),'isA',em['conceptnet']))
        yield pkg, conv['conv_id']
#conv 234             
if __name__ == "__main__":
    experiment('/code2/pel_test_data.jsonl')

"""{((0, 0), job_status, (7, 9)),
((0, 0), job_status, (7, 7)), 
((7, 7), 'isA', '/c/en/teacher'),
((7, 9), HasProperty, (10, 10)),
((7, 9), 'isA', 'c/en/teach')
((10, 10), 'isA', '/c/en/history'),




((0, 0), HasA, (11, 11)),
((11, 11), 'isA', '/c/en/kids'), 
((0, 0), HasProperty, (5, 5)), 
((5, 5), 'isA', '/c/en/romantic'), 
((0, 0), HasA, (1, 1)),
((1, 1), 'isA', '/c/en/antiques'),
((0, 0), school_status, (6, 6)),
((6, 6), 'isA', '/c/en/done'), 
 

((0, 0), HasA, (2, 2)),
((2, 2), 'isA', '/c/en/collection'),
((2, 2), HasProperty, (4, 4)),
((4, 4), 'isA', '/c/en/victorian')
((2, 2), HasProperty, (3, 3)), 
((3, 3), 'isA', '/c/en/doll'), 
}""" #conv 234
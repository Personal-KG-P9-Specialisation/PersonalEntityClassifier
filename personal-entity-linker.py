import difflib
from stanfordcorenlp import StanfordCoreNLP
import spacy
import neuralcoref

def stanford_coref(text):
    # RUN THIS FIRST FROM NLP FOLDER: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 15000 
    nlp = StanfordCoreNLP('http://localhost', port=9001)
    coref_chains = nlp.coref(text)
    print(coref_chains)
    nlp.close()
    return coref_chains

def neural_coref(text):
    nlp = spacy.load('en')
    neuralcoref.add_to_pipe(nlp)
    coref_chains = nlp(text)
    print(coref_chains._.coref_clusters)
    return coref_chains


def entityLinker(pkg, lookup, dialogue, utterance, mention, classification):   
    if classification == "pronoun":
        personalPronouns = ["my", "i", "our", "we", "us"]
        alternativePronouns = ["he", "she", "you", "it", "its", "they", "them", "their", "theirs", "his", "hers", "her", "him", "your"]

        if mention.lower() in personalPronouns:
            if mention.lower() in lookup.keys():
                return lookup[mention.lower()]
            else:
                for pronoun in personalPronouns:
                    if pronoun in lookup.keys():
                        lookup[mention.lower()] = lookup[pronoun]
                        return lookup[pronoun]
                # Next 3 lines might be removed - edge case: "existing" classification though not existing.
                idx = max(lookup.values()) + 1
                lookup[mention.lower()] = idx
                return idx

        elif mention.lower() in alternativePronouns:
            chains = neural_coref(dialogue + utterance)
            for chain in chains._.coref_clusters:
                for element in chain.mentions:
                    # TODO: Below check for span start and end should be fixed
                    if element.start == mention.start and element.end == mention.end:
                        entityMentions = [x for x in chain.mentions if x.lower() not in alternativePronouns]
                        match = difflib.get_close_matches(entityMentions[-1][-1], lookup.keys(), n=1)
                        if match[0].lower() in lookup.keys():
                            return lookup[match[0].lower()]
            # Next 3 lines might be removed - edge case: "existing" classification though not existing.
            idx = max(lookup.values()) + 1
            lookup[mention.lower()] = idx
            return idx

    elif classification == "new":
        idx = max(lookup.values()) + 1
        lookup[mention.lower()] = idx
        return idx

    elif classification == "existing":
        match = difflib.get_close_matches(mention, lookup.keys(), n=1)
        if match:
            if match[0].lower() is not mention:
                lookup[mention.lower()] = lookup[match[0].lower()]
                return lookup[match[0].lower()]
            elif match[0].lower() in lookup.keys():
                return lookup[match[0].lower()]
        # Next 3 lines might be removed - edge case: "existing" classification though not existing.
        idx = max(lookup.values()) + 1
        lookup[mention.lower()] = idx
        return idx

def testexample():
    pkg = [] # [[1, "desires", "c/en/sports"], [1, "attends", 2]]
    lookup = { "i":1 }
    dialogue = "" # "XXXXXXX"
    utterances = ["I like Obama. He is a great man. He can fly."]
    triples = [{"subject":"I", "relation":"like", "object":"Obama"}, {"subject":"Ohama", "relation":"IsA", "object":"c/en/man"}]

    for utterance in utterances:
        for triple in triples:
            subjectId = entityLinker(pkg, lookup, dialogue, utterance, triple["subject"], "existing")
            triple["subject"] = subjectId

            # We assume the hasValue and hasName relations always has a string literal as object. 
            if triple["relation"] != ("hasValue" or "hasName"):
                objectId = entityLinker(pkg, lookup, dialogue, utterance, triple["object"], "new")
                triple["object"] = objectId

            pkg.append(triple)
        dialogue = dialogue + utterance
    print(pkg)

if __name__ == "__main__":
    testexample()
    #spacy_coref("Hi I like Obama. He is a great man")
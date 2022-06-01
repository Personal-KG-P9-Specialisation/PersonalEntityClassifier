import difflib
from stanfordcorenlp import StanfordCoreNLP
import spacy
# import neuralcoref

def stanford_coref(text):
    # RUN THIS FIRST FROM NLP FOLDER: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 15000 
    nlp = StanfordCoreNLP('http://localhost', port=9001)
    coref_chains = nlp.coref(text)
    print(coref_chains)
    nlp.close()
    return coref_chains

# def spacy_coref(text):
    # nlp = spacy.load('en')
    # neuralcoref.add_to_pipe(nlp)
    # coref_chains = nlp(text)
    # return coref_chains

# Just for testing.
def get_classification(mention):
    if "c/en" in mention:
        return "string"
    else:
        return "existing"

def entityLinker(pkg, lookup, dialogue, utterance, mention, classification):    
    if classification == "new":
        idx = max(lookup.values()) + 1
        lookup[mention.lower()] = idx
        return idx

    elif classification == "existing":
        personalPronouns = ["my", "i"]
        alternativePronouns = ["he", "she", "you", "it", "we", "they", "their", "his", "her", "him", "your"]

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
            # TODO: Should consider entity mention's span start and end. Det er fint med bare at finde entity mention i tekst.
            # TODO: Should have dialogue history for both agents of conversation.
            # TODO: Should use entity linker from SpaCy
            chains = stanford_coref(dialogue + utterance)
            for chain in chains:
                for element in chain:
                    if element[-1].lower() == mention.lower():
                        entityMentions = [x for x in chain if x[-1].lower() not in alternativePronouns]
                        match = difflib.get_close_matches(entityMentions[-1][-1], lookup.keys(), n=1)
                        if match[0].lower() in lookup.keys():
                            return lookup[match[0].lower()]
            # Next 3 lines might be removed - edge case: "existing" classification though not existing.
            idx = max(lookup.values()) + 1
            lookup[mention.lower()] = idx
            return idx

        else:
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

    elif classification == "string":
        return mention

def test-example():
    pkg = [] # [[1, "desires", "c/en/sports"], [1, "attends", 2]]
    lookup = { "i":1 }
    dialogue = "" # "XXXXXXX"
    utterances = ["I like Obama. He is a great man. He can fly."]
    triples = [{"subject":"I", "relation":"like", "object":"Obama"}, {"subject":"Ohama", "relation":"IsA", "object":"c/en/man"}]

    for utterance in utterances:
        for triple in triples:
            subjectId = entityLinker(pkg, lookup, dialogue, utterance, triple["subject"], get_classification(triple["subject"]))
            triple["subject"] = subjectId

            # We assume the hasValue and hasName relations always has a string literal as object. 
            if triple["relation"] != ("hasValue" or "hasName"):
                objectId = entityLinker(pkg, lookup, dialogue, utterance, triple["object"], get_classification(triple["object"]))
                triple["object"] = objectId

            pkg.append(triple)
        dialogue = dialogue + utterance
    print(pkg)

if __name__ == "__main__":
    test-example()
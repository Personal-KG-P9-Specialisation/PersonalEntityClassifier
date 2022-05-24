import difflib
from stanfordcorenlp import StanfordCoreNLP

class User:
    def __init__(self):
        self.pkg = [] # [[1, "desires", "c/en/sports"], [1, "attends", 2]]
        self.lookup = { "i":1 }
        self.dialogue = "" # "XXXXXXX"

def coreference_resolution(text):
    # RUN THIS FIRST FROM NLP FOLDER: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 15000 
    nlp = StanfordCoreNLP('http://localhost', port=9001)
    coref_chains = nlp.coref(text)
    print(coref_chains)
    nlp.close()
    return coref_chains

# Just for testing.
def get_classification(mention):
    if "c/en" in mention:
        return "string"
    else:
        return "existing"

def entityLinker(user:User, utterance:str, mention:str, entityClassification:str):    
    if entityClassification == "new":
        idx = max(user.lookup.values()) + 1
        user.lookup[mention.lower()] = idx
        return idx

    elif entityClassification == "existing":
        personalPronouns = ["my", "i"]
        alternativePronouns = ["he", "she", "you", "it", "we", "they", "their", "his", "her", "him", "your"]

        if mention.lower() in personalPronouns:
            if mention.lower() in user.lookup.keys():
                return user.lookup[mention.lower()]
            else:
                for pronoun in personalPronouns:
                    if pronoun in user.lookup.keys():
                        user.lookup[mention.lower()] = user.lookup[pronoun]
                        return user.lookup[pronoun]
                # Next 3 lines might be removed - edge case: "existing" classification though not existing.
                idx = max(user.lookup.values()) + 1
                user.lookup[mention.lower()] = idx
                return idx

        elif mention.lower() in alternativePronouns:
            # TODO: Should consider entity mention's span start and end.
            # TODO: Should have dialogue history for both agents of conversation.
            chains = coreference_resolution(user.dialogue + utterance)
            for chain in chains:
                for element in chain:
                    if element[-1].lower() == mention.lower():
                        entityMentions = [x for x in chain if x[-1].lower() not in alternativePronouns]
                        match = difflib.get_close_matches(entityMentions[-1][-1], user.lookup.keys(), n=1)
                        if match[0].lower() in user.lookup.keys():
                            return user.lookup[match[0].lower()]
            # Next 3 lines might be removed - edge case: "existing" classification though not existing.
            idx = max(user.lookup.values()) + 1
            user.lookup[mention.lower()] = idx
            return idx

        else:
            match = difflib.get_close_matches(mention, user.lookup.keys(), n=1)
            if match:
                if match[0].lower() is not mention:
                    user.lookup[mention.lower()] = user.lookup[match[0].lower()]
                    return user.lookup[match[0].lower()]
                elif match[0].lower() in user.lookup.keys():
                    return user.lookup[match[0].lower()]
            # Next 3 lines might be removed - edge case: "existing" classification though not existing.
            idx = max(user.lookup.values()) + 1
            user.lookup[mention.lower()] = idx
            return idx

    elif entityClassification == "string":
        return mention

if __name__ == "__main__":
    user = User()

    utterances = ["I like Obama. He is a great man. He can fly."]
    triples = [{"subject":"I", "relation":"like", "object":"Obama"}, {"subject":"Ohama", "relation":"IsA", "object":"c/en/man"}]

    for utterance in utterances:
        for triple in triples:
            subjectId = entityLinker(user, utterance, triple["subject"], get_classification(triple["subject"]))
            triple["subject"] = subjectId
            objectId = entityLinker(user, utterance, triple["object"], get_classification(triple["object"]))
            triple["object"] = objectId
            user.pkg.append(triple)
        user.dialogue = user.dialogue + utterance

import spacy

nlp = spacy.load("en_core_web_lg")
tokens = nlp("dog Dog DOG chien")

for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
    print(token.vector[:20])


# For some reason something happened to my throat so i can't talk
# But this output makes sense according to the documentation

# Oh. are you alright?
# Yeah 
# OK. If you need to deal with that perhaps do that first before we continue to code.
import pickle

with open('wikidata.pickle', 'rb') as f:
    k = pickle.load(f)
    print(len(k))
    print(k[0])
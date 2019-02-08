import numpy as np

def unpack_deepwalk_embedding(filename):
    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format(filename, binary=False)
    features_matrix = np.asarray([model[str(node)] for node in range(1, model.vectors.shape[0]+1)])
    print('Embedding matrix shape: ', features_matrix.shape)
    return features_matrix

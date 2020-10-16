import hnswlib
import numpy as np


def create_index(data_path, name_path, index_path):
    data = np.load(data_path)
    data_labels = np.load(name_path)

    dim = 512
    num_elements = data.shape[0]
    print(num_elements)

    p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

    # Initing index
    # max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
    # during insertion of an element.
    # The capacity can be increased by saving/loading the index, see below.
    #
    # ef_construction - controls index search speed/build speed tradeoff
    #
    # M - is tightly connected with internal dimensionality of the data. Strongly affects memory consumption (~M)
    # Higher M leads to higher accuracy/run_time at fixed ef/efConstruction

    p.init_index(max_elements=num_elements, ef_construction=100, M=16)

    # Controlling the recall by setting ef:
    # higher ef leads to better accuracy, but slower search
    p.set_ef(10)

    # Element insertion (can be called several times):
    p.add_items(data, data_labels)

    p.save_index(index_path)

    del p


def load_index(index_path, dim):
    p = hnswlib.Index(space='l2', dim=dim)
    print("\nLoading index from '%s'\n" % index_path)
    p.load_index(index_path)
    return p

if __name__ == "__main__":
    create('faceEmbedding_IJBC.npy', 'name_IJBC.npy', 'IJBC_index.bin')
import hnswlib
import numpy as np

# class Index():
#     def __init__(self):
#         self.dim = 0
#         self.num_elements = 0

def create_index(data_path, name_path, index_path):
    data = np.load(data_path)
    data_labels = np.load(name_path)
    #print(type(data_labels))

    dim = 512
    num_elements = data.shape[0]
    # print(num_elements)

    p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip

    # Initing index
    # max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
    # during insertion of an element.
    # The capacity can be increased by saving/loading the index, see below.
    #
    # ef_construction - controls index search speed/build speed tradeoff
    #
    # M - is tightly connected with internal dimensionality of the data. Strongly affects memory consumption (~M)
    # Higher M leads to higher accuracy/run_time at fixed ef/efConstruction

    p.init_index(max_elements=num_elements, ef_construction=50, M=32)

    # Controlling the recall by setting ef:
    # higher ef leads to better accuracy, but slower search
    p.set_ef(25)

    # Element insertion (can be called several times):
    p.add_items(data, data_labels)

    p.save_index(index_path)
    del p


def load_index(index_path, dim):
    p = hnswlib.Index(space='cosine', dim=dim)
    print("\nLoading index from '%s'" % index_path)
    p.load_index(index_path)
    p.set_ef(50)
    return p

# def add_data(index_path, data, data_num):
#     p = hnswlib.Index(space='cosine', dim=dim)
#     p.load_index(index_path, max_elements=num_elements + data_num)
#     p.add_items(data)
#     p.set_ef(10)
#     p.save_index(index_path)
#     del p


if __name__ == "__main__":
    create_index('faceEmbedding_IJBC.npy', 'name_IJBC.npy', './database/IJB-C_index.bin')
    
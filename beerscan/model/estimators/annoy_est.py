from annoy import AnnoyIndex
import time

def annoy(img_descriptors, data_descriptors, metric="euclidean", trees=10, verbose=0):

    if verbose == 1:
        print("\n> ANNOY estimator")
        start_time = time.time()

    vec_dim = len(img_descriptors[0])
    annoy = AnnoyIndex(vec_dim, metric)
    for i, vect in enumerate(data_descriptors):

        if verbose == 1:
            print(f"\rBuilding ANNOY : {i}", end="")

        annoy.add_item(i, vect)
    annoy.build(trees)

    if verbose == 1:
        counter = 0
        print("\n")

    neighbors = []

    for vect in img_descriptors:

        if verbose == 1:
            print(f"\rcomputing {counter}", end=" ")
            counter += 1
        neighbors.append(annoy.get_nns_by_vector(vect, 5, search_k=-1, include_distances=False))

    if verbose == 1:
        print("\n--- %s seconds ---" % (time.time() - start_time))
    return neighbors

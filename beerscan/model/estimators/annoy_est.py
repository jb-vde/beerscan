from annoy import AnnoyIndex

import pandas as pd
import time


def build_from_df(vec_dim, descriptors_df:pd.DataFrame, metric="angular", n_trees=10,
                  verbose=False):

    if verbose:
        print("\n> Building ANNOY...")
        start_time = time.time()

    # TODO: Check descriptors_df format (columns match expectations)

    annoy = AnnoyIndex(vec_dim, metric)

    for index, row in descriptors_df.iterrows():

        if verbose:
            print(f"\rAdding vectors : {index}", end="")

        annoy.add_item(index, row["descriptor"])
    annoy.build(n_trees)

    if verbose:
        build_time = time.time() - start_time
        print(f"ANNOY built in {build_time} seconds")

    return annoy


def build_annoy(data_descriptors, metric="angular", n_trees=10, verbose=False,
                save_model=True):

    if verbose:
        print("\n> ANNOY estimator")
        start_time = time.time()

    vec_dim = len(data_descriptors[0])
    annoy = AnnoyIndex(vec_dim, metric)

    for i, vect in enumerate(data_descriptors):

        if verbose:
            print(f"\rBuilding ANNOY : {i}", end="")

        annoy.add_item(i, vect)

    annoy.build(n_trees)

    if verbose:
        build_time = time.time() - start_time
        print(f"ANNOY built in {build_time} seconds")

    if save_model:
        annoy.save('model.ann')

    return annoy


def run_annoy(img_descriptors, annoy, verbose=False):

    neighbors = []

    if verbose:
        print("\n> Start ANNOY search")
        start_time = time.time()

    for i, vect in enumerate(img_descriptors):

        if verbose:
            print(f"\rcomputing {i}", end=" ")
        neighbors.append(annoy.get_nns_by_vector(vect, 5, search_k=-1, include_distances=False))

    if verbose:
        run_time = time.time() - start_time
        print(f"ANNOY ran in {run_time} seconds")

    return neighbors


def load_annoy(vector_dim, model_path, metric="angular"):

    annoy = AnnoyIndex(vector_dim, metric=metric)
    annoy.load(model_path)

    return annoy




def annoy_from_data(img_descriptors, data_descriptors, metric="angular",
                    n_trees=10, verbose=False, save_model=True):

    annoy = build_annoy(data_descriptors, metric=metric,
                        n_trees=n_trees, verbose=verbose, save_model=save_model)

    neighbors = run_annoy(img_descriptors, annoy, verbose)

    return neighbors


def annoy_from_model(img_descriptors, verbose=False):

    vector_dim = len(img_descriptors[0])
    annoy = load_annoy(vector_dim)
    neighbors = run_annoy(img_descriptors, annoy, verbose)

    return neighbors

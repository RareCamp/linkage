import itertools

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


RANDOM_SEED = 0


def dummy(ref, p_flip):
    """
    Generaty a dummy bloom filter by flipping each bit with probability p_flip
    """
    return np.array([1 - bit if np.random.random() < p_flip else bit for bit in ref])


def purity(labels, ref_i, n_ref, n_dum):
    """
    Compute purity of the i-th reference bf cluster given all the labels
    """
    # j-th dummy bf for i-th reference bf
    dum_ij = lambda j: n_ref + ref_i * n_dum + j

    c = labels[ref_i]  # cluster of the i-th reference bf
    n_c = (labels == c).sum()  # number of records in c

    # i-th ref dummy records in c
    n_i_dum_c = (labels[dum_ij(0) : dum_ij(n_dum)] == c).sum()

    return n_i_dum_c / (n_dum + n_c - 1 - n_i_dum_c)


def find_k_star(X, n_D, B_ref, B_ref_dum):
    n_ref, n_dum = len(B_ref), len(B_ref_dum[0])

    max_purity, k_star = 0, 0
    for k in range(1, n_D + 1):
        # Fit k-means with k clusters and get labels
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init="auto").fit(X)

        # Compute purity
        purity_k = sum(purity(kmeans.labels_, i, n_ref, n_dum) for i in range(n_ref))

        # Update k* if purity is higher
        print(f"k={k: <2}  purity={purity_k:.2f}")
        if purity_k > max_purity:
            max_purity, k_star = purity_k, k

    print(f"\nk*={k_star}, purity={max_purity:.2f}")
    return k_star


def run():
    np.random.seed(RANDOM_SEED)

    fname = "data/sample.csv"
    # fname = "data/data100.csv"
    # fname = "data/data.csv"

    # Read input bloom filters
    df = pd.read_csv(fname, dtype="str")

    # Check that CSV file has column `bf`
    assert "bf" in df.columns, "CSV file doesn't have column `bf`"

    # Get bloom filter length and check that all bloom filters have the same length
    l = len(df.bf[0])
    assert all(df.bf.str.len() == l), "CSV file has different length of bloom filters"

    # Convert bloom filters to numpy arrays
    D = df.bf.apply(lambda x: np.array(list(map(int, list(x)))))

    # Set parameters
    n_ref = 20
    n_dum = 20
    p_flip = 0.2

    # Generate reference and dummy bloom filters
    # B_ref = [np.random.randint(0, 2, l) for _ in range(n_ref)]  # Method A
    B_ref = [D.sample().item() for _ in range(n_ref)]  # Method B
    B_ref_dum = [[dummy(ref, p_flip) for _ in range(n_dum)] for ref in B_ref]

    # Training data: reference bloom filters, dummy bloom filters, and data records
    X = B_ref + list(itertools.chain(*B_ref_dum)) + list(D)

    # Get optimal k
    k_star = find_k_star(X, len(D), B_ref, B_ref_dum)

    # Train k-means with optimal k
    kmeans = KMeans(n_clusters=k_star, random_state=RANDOM_SEED, n_init="auto").fit(X)

    # Add label column to dataset
    df["label"] = kmeans.labels_[n_ref + n_ref * n_dum :]

    # Save labeled dataset
    df[["id", "bf", "label"]].to_csv(fname.replace(".csv", "_labeled.csv"), index=False)

    print(df[["id", "bf", "label"]])


if __name__ == "__main__":
    run()

import itertools

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


from tokenizer import pii_tokenize

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
    # index of j-th dummy bf for i-th reference bf
    dum_ij_index = lambda j: n_ref + ref_i * n_dum + j

    c = labels[ref_i]  # cluster of the i-th reference bf
    n_c = (labels == c).sum()  # number of records in c

    # number of i-th ref dummy records in c
    n_i_dum_c = (labels[dum_ij_index(0) : dum_ij_index(n_dum)] == c).sum()

    return n_i_dum_c / (n_dum + n_c - 1 - n_i_dum_c)


def find_k_star(X, n_ref, n_dum, n_D):
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


def run(l=500, k=20, eps=2, p_flip=0.2):
    np.random.seed(RANDOM_SEED)

    fname = "voters.csv"
    df = pd.read_csv(fname, dtype="str")
    df = pd.concat([df, df], ignore_index=True)
    df["bf"] = df.apply(
        lambda row: pii_tokenize(
            l,
            k,
            eps,
            row["first_name"],
            row["middle_name"],
            row["last_name"],
            row["date_of_birth"],
            row["gender"],
        ),
        axis=1,
    )
    df = df.sort_values(by="first_name")

    if False:
        fname = "data/sample.csv"
        # fname = "data/data100.csv"
        # fname = "data/data.csv"

        # Read input bloom filters
        df = pd.read_csv(fname, dtype="str")

    # Check that CSV file has column `bf`
    assert "bf" in df.columns, "CSV file doesn't have column `bf`"

    # Get bloom filter length and check that all bloom filters have the same length
    l = len(df.bf[0])
    print("Bloom filter length:", l)
    assert all(df.bf.str.len() == l), "CSV file has different length of bloom filters"

    # Convert bloom filters to numpy arrays
    D = df.bf.apply(lambda x: np.array(list(map(int, list(x)))))

    # Set parameters
    n_ref = int(0.1 * len(D) + 1)
    n_dum = n_ref

    # Generate reference and dummy bloom filters
    # B_ref = [np.random.randint(0, 2, l) for _ in range(n_ref)]  # Method A
    B_ref = list(D.sample(n_ref))  # Method B
    B_ref_dum = [[dummy(ref, p_flip) for _ in range(n_dum)] for ref in B_ref]

    # Training data: reference bloom filters, dummy bloom filters, and data records
    X = B_ref + list(itertools.chain(*B_ref_dum)) + list(D)

    # Get optimal k
    k_star = find_k_star(X, n_ref, n_dum, len(D))

    # Train k-means with optimal k
    kmeans = KMeans(n_clusters=k_star, random_state=RANDOM_SEED, n_init="auto").fit(X)

    # Add label column to dataset
    df["label"] = kmeans.labels_[n_ref + n_ref * n_dum :]

    # Save labeled dataset
    df.to_csv(fname.replace(".csv", "_labeled.csv"), index=False)

    # Print labeled dataset
    print(df)

    # Compute false positive and false negative rates
    false_positives = sum(
        group.id.unique().size - 1 for _, group in df.groupby("label")
    )
    false_negatives = sum(
        group.label.unique().size - 1 for _, group in df.groupby("id")
    )
    print(
        f"False positives rate: {false_positives / len(df):.2f} (two unrelated patients with the same label)"
    )
    print(
        f"False negatives rate: {false_negatives / len(df):.2f} (same patient with different labels)"
    )


if __name__ == "__main__":
    run(1000, eps=2)

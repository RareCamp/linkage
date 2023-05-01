import collections
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
    for k in range(1, n_D + 1, 10):
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


def false_positives(df):
    """
    Find number of people who are wrongly matched to someone else
    """
    for label, g in df.groupby("label"):
        if g.id.unique().size > 1:
            print("\nFALSE POSITIVE:")
            print(g)
    return sum(len(g) for _, g in df.groupby("label") if g.id.unique().size > 1)


def false_negatives(df):
    """
    Find number of records from the same person that are not matched
    """
    fns = 0
    for _, group in df.groupby("id"):
        main_cluster = collections.Counter(group["label"]).most_common(1)[0][0]
        #print(f"main_cluster={main_cluster}")
        if (group["label"] != main_cluster).sum() > 0:
            print("\nFALSE NEGATIVE:")
            print(group)
        fns += (group["label"] != main_cluster).sum()
    print(fns)
    return fns


def run(l=500, k=20, eps=2, p_flip=0.0):
    np.random.seed(RANDOM_SEED)

    fname = "ncvr_numrec_100_modrec_2_ocp_0.csv"
    df = pd.read_csv(fname, dtype="str")
    print(df)
    df["bf"] = df.apply(
        lambda row: pii_tokenize(
            l,
            k,
            eps,
            row["first_name"],
            row["last_name"],
            row["city"],
            row["date_of_birth"],
            row["gender"],
        ),
        axis=1,
    )
    df = df.sort_values(by="id")

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

    # Print false positive and false negative rates
    fpr = 100*false_positives(df) / len(df)
    fnr = 100*false_negatives(df) / len(df)

    print(f"\nk*: {k_star}")
    print(f"False positive rate: {fpr:.2f} %")
    print(f"False negative rate: {fnr:.2f} %")


if __name__ == "__main__":
    run(l=1000, k=20, eps=100, p_flip=0.00)

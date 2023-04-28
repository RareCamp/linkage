from hashlib import sha256
from math import exp

import numpy as np


def tokenize(l, k, eps, fields):
    bf = [0]*l
    for field in fields:
        for i in range(k):
            bf[int(sha256(f"{field}#{i}".encode("utf-8")).hexdigest(), 16) % l] = 1
    # return bf
    eta = 1. - 1. / (1. + exp(eps))
    return [bit if np.random.random() <= eta else 1 - bit for bit in bf]


def q_grams(s, q=2, prefix=""):
    s = "".join(filter(str.isalpha, s.lower()))
    return [prefix + s[i : i + q] for i in range(len(s) - q + 1)]

def pii_tokenize(l, k, eps, first_name, middle_name, last_name, date_of_birth, gender):
    fields = (
        q_grams(first_name, prefix="fname:") +
        q_grams(middle_name, prefix="mname:") +
        q_grams(last_name, prefix="lname:") +
        ["dob:" + date_of_birth, "gender:" + gender]
    )
    print(fields)
    bf =  tokenize(l, k, eps, fields)
    return "".join(map(str, bf))

import math
import re
from collections import defaultdict
from functools import partial
from itertools import combinations
from multiprocessing import Pool, cpu_count

import nltk
import numpy as np
from nltk.util import ngrams
from sklearn.manifold import MDS
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


# Get the k-beginning and k-end of each n-gram
def compute_beginning_end(ngram: list[str], k: int) -> tuple[tuple[str]]:
    beginning = ngram[:k]
    end = ngram[-k:]
    context = (beginning, end)
    return context


# Define a nested defaultdict to store the n-gram counts
def count_ngrams(corpus: list[str], n: int, k: int):
    # Count the n-grams and their individual contexts
    _context_counts = defaultdict(int)
    _ngram_counts = defaultdict(int)
    for ngram in tqdm(ngrams(corpus, n), desc="Counting n-grams"):
        context = compute_beginning_end(ngram, k)
        _ngram_counts[ngram] += 1
        _context_counts[context] += 1

    return _ngram_counts, _context_counts


def compute_log_quotient(ngram_counts, context_counts, k):
    # Compute the logarithms of the quotients and store in the nested dictionary
    middle_words = []
    dd_float = partial(defaultdict, float)
    _log_quotient_counts = defaultdict(dd_float)
    for ngram, count in tqdm(ngram_counts.items(), desc="Computing log quotients"):
        context = compute_beginning_end(ngram, k)
        middle_words.append(ngram[k])
        quotient = count / context_counts[context]
        log_quotient = math.log(quotient)
        _log_quotient_counts[context][ngram] = log_quotient

    middle_words = sorted(set(middle_words))
    return _log_quotient_counts, middle_words


def compute_distance_commands(_combination, _log_quotient_counts, _middle_words, _context, _k):
    results = []
    ngram1, ngram2 = _combination
    if ngram1 != ngram2:
        _abs_diff = abs(
            _log_quotient_counts[_context][ngram1]
            - _log_quotient_counts[_context][ngram2]
        )
        _x = _middle_words.index(ngram1[_k])
        _y = _middle_words.index(ngram2[_k])
        results.append((_x, _y, _abs_diff))
    return results


def compute_distances(item, _log_quotient_counts, _middle_words, _k):
    print(".", end="")
    _context, _ngrams = item
    if len(_ngrams) <= 1:
        return []
    combination = combinations(_ngrams.keys(), 2)
    results = []
    for c in combination:
        results += compute_distance_commands(c, _log_quotient_counts, _middle_words, _context, _k)
    return results


def compute_distance_matrix(_log_quotient_counts, _middle_words, k):
    matrix_results = []
    partial_compute_distance = partial(compute_distances, _log_quotient_counts=_log_quotient_counts,
                                       _middle_words=_middle_words, _k=k)
    with Pool(cpu_count()) as p:
        partial_results = p.map(partial_compute_distance, _log_quotient_counts.items(), chunksize=1000)
    #partial_results = process_map(partial_compute_distance, _log_quotient_counts.items(), max_workers=cpu_count() * 2,
    #                              chunksize=1000)
        matrix_results += partial_results

    return matrix_results


def remove_rows_columns(_matrix, _indices):
    # Remove specified rows
    _matrix = np.delete(_matrix, _indices[0], axis=0)

    # Remove specified columns
    _matrix = np.delete(_matrix, _indices[1], axis=1)

    return _matrix


def matrix_filtration(_matrix):
    # Find rows and columns with only NaN and 0 values
    nan_zero_rows = np.all(np.isnan(_matrix) | (_matrix == 0), axis=1)
    nan_zero_cols = np.all(np.isnan(_matrix) | (_matrix == 0), axis=0)

    # Get the indices of removed rows and columns
    _removed_rows = np.where(nan_zero_rows)[0]
    _removed_cols = np.where(nan_zero_cols)[0]

    # Remove rows and columns with only NaN and 0 values
    _filtered_matrix = _matrix[~nan_zero_rows, :]
    _filtered_matrix = _filtered_matrix[:, ~nan_zero_cols]

    return _filtered_matrix, _removed_rows, _removed_cols


def cluster_decompostion(matrix):
    # Iterate over the matrix elements
    for idx, val in np.ndenumerate(matrix):
        i, j = idx
        if j > i and np.isnan(val):
            print("Found nan value at position (i={}, j={})".format(i, j))
            break
    _principal_matrix = np.array(matrix[0:j,
                                 0:j])
    _secondary_matrix = np.array(matrix[j:len(matrix) + 1,
                                 j:len(matrix) + 1])
    _cluster_relation_matrix = np.array(matrix[0:j,
                                        j:len(matrix) + 1])

    return _principal_matrix, _secondary_matrix, _cluster_relation_matrix


def prob_matrix(_matrix):
    _prob_matrix = -np.exp(_matrix)
    _processed_prob_matrix = np.nan_to_num(_prob_matrix, nan=0)
    return _processed_prob_matrix


def apply_commands(_commands, _distance_matrix):
    for x, y, abs_diff in tqdm(_commands, desc="Filling distance matrix"):
        if not np.isnan(_distance_matrix[x, y]):
            _distance_matrix[x, y] = max(_distance_matrix[x, y], abs_diff)
        else:
            _distance_matrix[x, y] = abs_diff
        if not np.isnan(_distance_matrix[y, x]):
            _distance_matrix[y, x] = max(_distance_matrix[y, x], abs_diff)
        else:
            _distance_matrix[y, x] = abs_diff
    return _distance_matrix


def initialize_data(k=2, corpus_name="brown"):
    corpus_dict = {
        "brown": nltk.corpus.brown,
        "gutenberg": nltk.corpus.gutenberg,
        "reuters": nltk.corpus.reuters,
        "inaugural": nltk.corpus.inaugural,
        "webtext": nltk.corpus.webtext,
        "nps_chat": nltk.corpus.nps_chat,
        "abc": nltk.corpus.abc,
        "genesis": nltk.corpus.genesis,
        "state_union": nltk.corpus.state_union,
        "treebank": nltk.corpus.treebank,
        "wordnet": nltk.corpus.wordnet,
        "movie_reviews": nltk.corpus.movie_reviews,
        "conll2000": nltk.corpus.conll2000,
        "conll2002": nltk.corpus.conll2002,
        "semcor": nltk.corpus.semcor,
        "floresta": nltk.corpus.floresta,
        "indian": nltk.corpus.indian,
        "mac_morpho": nltk.corpus.mac_morpho,
        "cess_cat": nltk.corpus.cess_cat,
        "cess_esp": nltk.corpus.cess_esp,
        "udhr": nltk.corpus.udhr,
        "tagged_treebank_para_block_reader": nltk.corpus.tagged_treebank_para_block_reader,
        "sinica_treebank": nltk.corpus.sinica_treebank,
        "alpino": nltk.corpus.alpino,
        "comparative_sentences": nltk.corpus.comparative_sentences,
        "dependency_treebank": nltk.corpus.dependency_treebank,
    }

    # Download the Brown corpus
    nltk.download(corpus_name)
    # Load the corpus
    # corpus_raw = nltk.corpus.brown.words()
    corpus_raw = corpus_dict[corpus_name].words()
    corpus_raw = [word.lower() for word in corpus_raw]
    # corpus = nltk.corpus.CorpusReader(nltk.data.find(f"corpora/{corpus_name}"), ".*")
    # corpus_raw = nltk.tokenize.word_tokenize(corpus.raw())
    print(f"Corpus size: {len(corpus_raw)}")
    corpus = []
    for word in tqdm(corpus_raw, desc="Pre-processing corpus"):
        if word is not None:
            if re.match(r'^[A-Za-z"\']', word):
                word = word.replace(",", "")
                corpus.append(word)
    # Set the value of n for n-grams
    n = 2 * k + 1
    ngram_counts, context_counts = count_ngrams(corpus, n, k)
    _log_quotient_counts, _middle_words = compute_log_quotient(ngram_counts, context_counts, k)

    return _log_quotient_counts, _middle_words


def write_distance_commands(log_quotient_counts, middle_words, k=2):
    print(f"Computing distances. (k={k})")
    # Compute the maximum absolute differences between all combinations of ngrams with the same context
    distance_matrix_commands = compute_distance_matrix(log_quotient_counts, middle_words, k)
    commands = []
    for line in tqdm(distance_matrix_commands, desc="Parsing distance matrix commands"):
        line = line.replace("[]", "")
        # parse line of python lists
        line = line.replace("[", "").replace("]", "\n").replace("), (", ")\n(")
        commands.append(line)
    with open("commands.txt", "w") as f:
        f.writelines(commands)


def write_similarity_matrix(_middle_words):
    size = len(_middle_words)
    distance_matrix = np.empty((size, size,))
    distance_matrix[:] = np.nan
    # set diagonal to 0
    np.fill_diagonal(distance_matrix, 0)

    with open("commands.txt") as f:
        commands = f.readlines()
    commands = [eval(x) for x in commands]
    distance_matrix = apply_commands(commands, distance_matrix)
    # print numpy matrix memory usage (in Gb)
    print(f"Memory usage of the distance matrix: {distance_matrix.nbytes / 1024 ** 3} Gb")
    print(f"Shape of the distance matrix: {distance_matrix.shape}")
    # Remove rows and columns with only NaN and 0 values
    distance_matrix, removed_rows, removed_cols = matrix_filtration(distance_matrix)
    # print numpy matrix memory usage (in Gb)
    print(f"Memory usage of the distance matrix: {distance_matrix.nbytes / 1024 ** 3} Gb")
    print(f"Shape of the distance matrix: {distance_matrix.shape}")
    with open("matrix.txt", "w") as f:
        np.savetxt(f, distance_matrix, fmt="%s")
    print("All entries of the matrix are NAN: ", np.isnan(distance_matrix).all())
    similarity_matrix = prob_matrix(distance_matrix)
    # assert any value in the matrix is not in [-1,  0]
    print("All entries are 0 or -1", not (np.any(similarity_matrix != -1) and np.any(similarity_matrix != 0)))
    print(similarity_matrix)
    # write similarity matrix to file
    with open("similarity_matrix.txt", "w") as f:
        np.savetxt(f, similarity_matrix, fmt="%s")


def compute_embedding(n=100):
    # load similarity matrix from file
    with open("similarity_matrix.txt") as f:
        similarity_matrix = np.loadtxt(f)
    # Define dimension of the embedding
    mds = MDS(n_components=n, metric=False, normalized_stress=True, dissimilarity='precomputed')
    X_mds = mds.fit_transform(similarity_matrix)
    stress = mds.stress_
    print(f"This is the Stress of the algorithm: {stress}")


if __name__ == "__main__":
    log_quotient_counts, middle_words = initialize_data()

    write_distance_commands(log_quotient_counts, middle_words)

    write_similarity_matrix(middle_words)

    compute_embedding()

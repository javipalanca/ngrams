import numpy as np

from cat_dist import (
    count_ngrams,
    compute_beginning_end,
    compute_log_quotient,
    compute_distance_matrix, matrix_filtration, prob_matrix, apply_commands,
)


def test_compute_beginning_end():
    ngram = ("the", "red", "fox", "jumps", "the", "tree")
    k = 1
    assert compute_beginning_end(ngram, k) == (("the",), ("tree",))
    k = 2
    assert compute_beginning_end(ngram, k) == (("the", "red"), ("the", "tree"))
    k = 3
    assert compute_beginning_end(ngram, k) == (
        ("the", "red", "fox"),
        ("jumps", "the", "tree"),
    )


def test_count_ngrams_k1():
    corpus = ["the", "red", "fox", "jumps", "the", "tree"]
    k = 1
    n = 3

    ngram_counts, context_counts = count_ngrams(corpus, n, k)

    assert ngram_counts == {
        ("the", "red", "fox"): 1,
        ("red", "fox", "jumps"): 1,
        ("fox", "jumps", "the"): 1,
        ("jumps", "the", "tree"): 1,
    }

    assert context_counts == {
        (("the",), ("fox",)): 1,
        (("red",), ("jumps",)): 1,
        (("fox",), ("the",)): 1,
        (("jumps",), ("tree",)): 1,
    }


def test_count_ngrams_k2():
    corpus = ["the", "red", "fox", "jumps", "the", "tree"]
    k = 2
    n = 3

    ngram_counts, context_counts = count_ngrams(corpus, n, k)

    assert ngram_counts == {
        ("the", "red", "fox"): 1,
        ("red", "fox", "jumps"): 1,
        ("fox", "jumps", "the"): 1,
        ("jumps", "the", "tree"): 1,
    }

    assert context_counts == {
        (
            ("the", "red"),
            (
                "red",
                "fox",
            ),
        ): 1,
        (
            ("red", "fox"),
            (
                "fox",
                "jumps",
            ),
        ): 1,
        (
            ("fox", "jumps"),
            (
                "jumps",
                "the",
            ),
        ): 1,
        (
            ("jumps", "the"),
            (
                "the",
                "tree",
            ),
        ): 1,
    }


def test_count_ngrams_more_contexts():
    corpus = [
        "the",
        "red",
        "fox",
        "jumps",
        "the",
        "tree",
        "the",
        "quick",
        "fox",
        "jumps",
        "the",
        "lazy",
        "dog",
    ]
    k = 1
    n = 3

    ngram_counts, context_counts = count_ngrams(corpus, n, k)

    assert ngram_counts == {
        ("the", "red", "fox"): 1,
        ("red", "fox", "jumps"): 1,
        ("fox", "jumps", "the"): 1,
        ("jumps", "the", "tree"): 1,
        (
            "the",
            "tree",
            "the",
        ): 1,
        ("tree", "the", "quick"): 1,
        ("the", "quick", "fox"): 1,
        ("quick", "fox", "jumps"): 1,
        ("fox", "jumps", "the"): 2,
        ("jumps", "the", "lazy"): 1,
        ("the", "lazy", "dog"): 1,
    }

    assert context_counts == {
        (("the",), ("fox",)): 2,
        (("red",), ("jumps",)): 1,
        (("fox",), ("the",)): 2,
        (("jumps",), ("tree",)): 1,
        (("the",), ("the",)): 1,
        (("tree",), ("quick",)): 1,
        (("quick",), ("jumps",)): 1,
        (("jumps",), ("lazy",)): 1,
        (("the",), ("dog",)): 1,
    }


def test_compute_log_quotient():
    ngram_counts = {
        ("the", "red", "fox"): 1,
    }
    context_counts = {
        (("the",), ("fox",)): 1,
    }

    result, middle_words = compute_log_quotient(ngram_counts, context_counts, 1)
    assert result == {(("the",), ("fox",)): {("the", "red", "fox"): 0}}

    assert middle_words == ["red"]


def test_distance_matrix():
    log_quotient = {
        (("the",), ("fox",)): {
            ("the", "red", "fox"): 0,
            ("the", "quick", "fox"): 0.5,
        }
    }

    commands = compute_distance_matrix(log_quotient, ["red", "quick"], 1)
    # flatten list of lists
    commands = [item for sublist in commands for item in sublist]

    assert commands == [(0, 1, 0.5)]

    result = np.empty((2, 2,))
    result[:] = np.nan
    np.fill_diagonal(result, 0)

    result = apply_commands(commands, result)

    assert np.array_equal(result, np.array([[0, 0.5], [0.5, 0]]))


def test_distance_matrix_notbalanced():
    log_quotient = {
        (("the",), ("fox",)): {
            ("the", "red", "fox"): 1,
            ("the", "quick", "fox"): 2,
        },
        (("a",), ("rabbit",)): {
            ("a", "blue", "rabbit"): 3,
            ("a", "yellow", "rabbit"): 4,
            ("a", "black", "rabbit"): 5,
        }
    }

    middle_words = ["black", "blue", "quick", "red", "yellow"]
    commands = compute_distance_matrix(log_quotient, middle_words, 1)

    # flatten list of lists
    commands = [item for sublist in commands for item in sublist]

    assert commands == [(3, 2, 1), (1, 4, 1), (1, 0, 2), (4, 0, 1)]

    result = np.empty((5, 5,))
    result[:] = np.nan
    np.fill_diagonal(result, 0)

    result = apply_commands(commands, result)

    assert np.array_equal(result,
                          np.array([
                              [0., 2., np.nan, np.nan, 1.],
                              [2., 0., np.nan, np.nan, 1.],
                              [np.nan, np.nan, 0., 1., np.nan],
                              [np.nan, np.nan, 1., 0., np.nan],
                              [1., 1., np.nan, np.nan, 0.]
                          ]
                          ), equal_nan=True
                          )


def test_matrix_filtration_returns_correct_type():
    matrix = np.array([[1, 2], [3, 4]])
    result = matrix_filtration(matrix)
    assert isinstance(result, tuple)


def test_matrix_filtration_returns_correct_matrix():
    matrix = np.array([[0, 0, np.nan],
                       [0, np.nan, 3],
                       [4, 0, 6]])
    expected_result = np.array([[0, 3],
                                [4, 6]])
    result = matrix_filtration(matrix)[0]
    np.testing.assert_array_equal(result, expected_result)


def test_matrix_filtration_returns_same_matrix():
    matrix = np.array([[1, 0, np.nan],
                       [0, np.nan, 3],
                       [4, 5, 6]])
    expected_result = np.array([[1, 0, np.nan],
                                [0, np.nan, 3],
                                [4, 5, 6]])
    result = matrix_filtration(matrix)[0]
    np.testing.assert_array_equal(result, expected_result)


def test_matrix_filtration_returns_correct_removed_rows_and_cols():
    matrix = np.array([[0, 0, np.nan],
                       [0, np.nan, 3],
                       [4, 0, 6]])
    expected_removed_rows = np.array([0])
    expected_removed_cols = np.array([1])
    result_removed_rows = matrix_filtration(matrix)[1]
    result_removed_cols = matrix_filtration(matrix)[2]
    np.testing.assert_array_equal(result_removed_rows, expected_removed_rows)
    np.testing.assert_array_equal(result_removed_cols, expected_removed_cols)


def test_matrix_filtration_returns_all_removed_rows_and_cols():
    matrix = np.array([[0, 0, np.nan],
                       [0, np.nan, 0],
                       [0, 0, 0]])
    expected_removed_rows = np.array([0, 1, 2])
    expected_removed_cols = np.array([0, 1, 2])
    result_removed_rows = matrix_filtration(matrix)[1]
    result_removed_cols = matrix_filtration(matrix)[2]
    np.testing.assert_array_equal(result_removed_rows, expected_removed_rows)
    np.testing.assert_array_equal(result_removed_cols, expected_removed_cols)


def test_prob_matrix2x2():
    # Test con una matriz de 2x2
    matrix_2x2 = np.array([[1, 2],
                           [3, 4]])
    expected_2x2 = np.array([[-2.71828183, -7.3890561],
                             [-20.08553692, -54.59815003]])
    np.testing.assert_allclose(prob_matrix(matrix_2x2), expected_2x2)


def test_prob_matrix3x3_with_nan_and_0():
    # Test con una matriz de 3x3 con NaN
    matrix_3x3 = np.array([[1, 2, np.nan],
                           [3, np.nan, 5],
                           [np.nan, 7, 0]])
    expected_3x3 = np.array([[-2.71828183, -7.3890561, 0.],
                             [-20.08553692, 0., -148.41315910], [0., -1096.63315842, -1]])
    np.testing.assert_allclose(prob_matrix(matrix_3x3), expected_3x3)


def test_prob_matrix4x4_with_big_values():
    # Test con una matriz de 4x4 con valores grandes
    matrix_4x4 = np.array([[100, 200, 300, 400],
                           [900, 1000, 1100, 1200],
                           [1300, 1400, 1500, 1600]])
    expected_4x4 = np.array([[-2.688117e43, -7.225974e086, -1.942426e130, -5.221470e173],
                             [-1.797693e308, -1.797693e308, -1.797693e308, -1.797693e308],
                             [-1.797693e308, -1.797693e308, -1.797693e308, -1.797693e308]])
    np.testing.assert_allclose(prob_matrix(matrix_4x4), expected_4x4, rtol=1e-6)

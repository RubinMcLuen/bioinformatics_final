# These are the similarity measure functions that are called in main.py
# I tried to make them follow the formula as clearly as possible.

import numpy as np

def manhattan_distance(e_f, e_g):
    return np.sum(np.abs(e_f - e_g))

def euclidean_distance(e_f, e_g):
    return np.sqrt(np.sum((e_f - e_g) ** 2))

def mahalanobis_distance(e_f, e_g, cov_matrix, epsilon=1e-5):
    cov_matrix_reg = cov_matrix + np.eye(cov_matrix.shape[0]) * epsilon
    diff = e_f - e_g
    return np.dot(np.dot(diff, np.linalg.inv(cov_matrix_reg)), diff.T)

def pearson_correlation(e_f, e_g):
    e_f_mean = np.mean(e_f)
    e_g_mean = np.mean(e_g)
    numerator = np.sum((e_f - e_f_mean) * (e_g - e_g_mean))
    denominator = np.sqrt(np.sum((e_f - e_f_mean) ** 2) * np.sum((e_g - e_g_mean) ** 2))
    r_fg = numerator / denominator
    return 1 - r_fg

def uncentered_correlation(e_f, e_g):
    numerator = np.sum(e_f * e_g)
    denominator = np.sqrt(np.sum(e_f ** 2) * np.sum((e_g ** 2)))
    r_fg = numerator / denominator
    return 1 - r_fg

def spearman_rank_correlation(e_f, e_g):
    e_f_ranks = np.argsort(e_f) + 1
    e_g_ranks = np.argsort(e_g) + 1
    e_f_mean = np.mean(e_f_ranks)
    e_g_mean = np.mean(e_g_ranks)
    numerator = np.sum((e_f_ranks - e_f_mean) * (e_g_ranks - e_g_mean))
    denominator = np.sqrt(np.sum((e_f_ranks - e_f_mean) ** 2) * np.sum((e_g_ranks - e_g_mean) ** 2))
    r_fg = numerator / denominator
    return 1 - r_fg

def absolute_correlation(e_f, e_g):
    # Pearson correlation first
    e_f_mean = np.mean(e_f)
    e_g_mean = np.mean(e_g)
    numerator = np.sum((e_f - e_f_mean) * (e_g - e_g_mean))
    denominator = np.sqrt(np.sum((e_f - e_f_mean) ** 2) * np.sum((e_g - e_g_mean) ** 2))
    r_fg = numerator / denominator

    return 1 - abs(r_fg)

def squared_correlation(e_f, e_g):
    # Pearson correlation first
    e_f_mean = np.mean(e_f)
    e_g_mean = np.mean(e_f)
    numerator = np.sum((e_f - e_f_mean) * (e_g - e_g_mean))
    denominator = np.sqrt(np.sum((e_f - e_f_mean) ** 2) * np.sum((e_g - e_g_mean) ** 2))
    r_fg = numerator / denominator

    return 1 - (r_fg ** 2)
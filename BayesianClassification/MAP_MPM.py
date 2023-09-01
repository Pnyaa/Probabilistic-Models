#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from scipy.stats import norm
import time


def configure_logging():
    logging.basicConfig(style='{', format="{asctime} : {message}", datefmt="%c", level=logging.INFO)


def calc_probaprio2(X, cl1, cl2):
    p1 = X[X == cl1].size / X.size
    p2 = X[X == cl2].size / X.size
    return [p1, p2]


def bruit_gauss2(X, cl1, cl2, m1, sig1, m2, sig2):
    bruit1 = np.random.normal(m1, sig1, X.shape)
    bruit2 = np.random.normal(m2, sig2, X.shape)
    Y = X.copy().astype(float)
    Y[X == cl1] = bruit1[X == cl1]
    Y[X == cl2] = bruit2[X == cl2]
    return Y


def classif_gauss2(Y, cl1, cl2, m1, sig1, m2, sig2):
    probas = np.zeros((Y.shape[0], Y.shape[1], 2))
    probas[:, :, 0] = norm.pdf(Y, loc=m1, scale=sig1)
    probas[:, :, 1] = norm.pdf(Y, loc=m2, scale=sig2)
    S = probas.argmax(axis=2)
    S[S == 0] = cl1
    S[S == 1] = cl2
    return S


def MAP_MPM2(Y, cl1, cl2, p1, p2, m1, sig1, m2, sig2):
    probas = np.zeros((Y.shape[0], Y.shape[1], 2))
    probas[:, :, 0] = p1 * norm.pdf(Y, loc=m1, scale=sig1)
    probas[:, :, 1] = p2 * norm.pdf(Y, loc=m2, scale=sig2)
    S = probas.argmax(axis=2)
    S[S == 0] = cl1
    S[S == 1] = cl2
    return S


def taux_erreur(A, B):
    errors = A[A != B]
    return errors.size / A.size


def taux_erreur_moyen(X, cl1, cl2, p1, p2, m1, sig1, m2, sig2, nombre_signaux, function):
    # Concatène nombre_signaux dans le vecteur X pour paralléliser la génération des vecteurs bruités et classifiés
    concat_X = np.tile(X, (1, nombre_signaux)).T
    Y = bruit_gauss2(concat_X, cl1, cl2, m1, sig1, m2, sig2)
    if function == "MAP":
        S = MAP_MPM2(Y, cl1, cl2, p1, p2, m1, sig1, m2, sig2)
    else:
        S = classif_gauss2(Y, cl1, cl2, m1, sig1, m2, sig2)
    return taux_erreur(concat_X, S)


def plot_XYS(X, cl1, cl2, p1, p2, m1, sig1, m2, sig2):
    Y = bruit_gauss2(X, cl1, cl2, m1, sig1, m2, sig2)
    S = MAP_MPM2(Y, cl1, cl2, p1, p2, m1, sig1, m2, sig2)
    plt.plot(X, label="Signal original")
    plt.plot(S, label="Signal segmenté", linestyle='dashed')
    plt.plot(Y, label="Signal bruité", color="limegreen")
    plt.xlabel("N° échantillon")
    plt.ylabel("Valeur signal")
    plt.title(f"Reconstruction du signal pour m1={m1}, m2={m2}, sig1={sig1} et sig2={sig2}")
    plt.legend(loc="upper right")
    plt.draw()
    plt.pause(1)


def mean_rate(X, cl1, cl2, p1, p2, m1, sig1, m2, sig2, iterations, function):
    params = [(X, cl1, cl2, p1, p2, m1, sig1, m2, sig2, i, function) for i in range(1, iterations + 1)]
    with multiprocessing.get_context("spawn").Pool() as p:
        mean_rates = p.starmap(taux_erreur_moyen, params)
    ultimate_mean = sum(mean_rates) / len(mean_rates)
    x = np.arange(0, len(mean_rates))
    plt.figure(2, figsize=(8, 6), dpi=80)
    plt.plot(x, mean_rates)
    plt.plot(x, ultimate_mean * np.ones(iterations), color="red", label=f"Erreur moyenne : {ultimate_mean:.3e}")
    plt.title(f"Evolution du taux d'erreur moyen pour m1={m1}, m2={m2}, sig1={sig1} et sig2={sig2}")
    plt.xlabel("Nombre de signaux sur lesquels l'erreur est moyennée")
    plt.ylabel("Taux d'erreur moyen")
    plt.legend(loc="upper right")
    plt.draw()
    plt.pause(1)
    return ultimate_mean


def simul2(N, cl1, cl2, p1, p2):
    if p1 + p2 != 1:
         raise ValueError("Probabilities do not sum to 1")
    nb_cl1 = round(p1 * N)
    signal = np.concatenate((cl1 * np.ones(nb_cl1, dtype=int), cl2 * np.ones(N - nb_cl1, dtype=int)))
    np.random.shuffle(signal)
    return signal


def main():
    # Bug avec multiprocessing : https://pythonspeed.com/articles/python-multiprocessing/
    multiprocessing.set_start_method("spawn")
    configure_logging()
    m1, m2 = [120, 127, 127, 127, 127], [130, 127, 128, 128, 128]
    sig1, sig2 = [1, 1, 1, 0.1, 2], [2, 5, 1, 0.1, 3]

    # Question 1 & 2
    # signaux = ["signal", "signal1", "signal2", "signal3", "signal4", "signal5"]
    # iterations = [2000, 1000, 500, 500, 500, 500]
    # for signal in range(len(signaux)):
    #     X = np.load(f"./signaux/{signaux[signal]}.npy")
    #     X = X.reshape((X.shape[0], 1))
    #     cl = np.unique(X)
    #     cl1, cl2 = cl[0], cl[1]
    #     probaprio = calc_probaprio2(X, cl1, cl2)
    #     logging.info(probaprio)
    #     t = time.time()
    #     for i in range(len(m1)):
    #         plt.figure(1)
    #         plot_XYS(X, cl1, cl2, probaprio[0], probaprio[1], m1[i], sig1[i], m2[i], sig2[i])
    #         plt.savefig(f"figures/MAP/{signaux[signal]}/XYS_{i + 1}.png")
    #         plt.close()
    #         ultimate_mean = mean_rate(X, cl1, cl2, probaprio[0], probaprio[1], m1[i], sig1[i], m2[i], sig2[i], iterations[signal], "MAP")
    #         logging.info(f"Mean error rate for {signaux[signal]}.npy with up to {iterations[signal]} signals for m1={m1[i]}, m2={m2[i]}, sig1={sig1[i]}, sig2={sig2[i]} : {ultimate_mean}")
    #         plt.savefig(f"figures/MAP/{signaux[signal]}/Mean_error_rate_{i + 1}.png")
    #         plt.close()
    #     logging.info(f"Time taken for {signaux[signal]}.npy : {time.time() - t}")

    # Question 3&4
    cl1, cl2 = 100, 200
    probas = [0.1, 0.3, 0.5, 0.7, 0.9]
    signaux = [simul2(500, cl1, cl2, i, 1-i) for i in probas]
    for i in range(len(signaux)):
        X = signaux[i].reshape((signaux[i].shape[0], 1))
        p1 = probas[i]
        p2 = 1 - p1
        for j in range(len(m1)):
            ultimate_mean = mean_rate(X, cl1, cl2, p1, p2, m1[j], sig1[j], m2[j], sig2[j], 1000, "MAP")
            logging.info(f"Mean error rate for p1={p1} with up to 1000 signals with MAP algo for m1={m1[j]}, m2={m2[j]}, sig1={sig1[j]}, sig2={sig2[j]} : {ultimate_mean}")
            plt.savefig(f"figures/simul/Mean_error_rate_MAP_p{p1}_{j+1}.png")
            plt.close()
            ultimate_mean = mean_rate(X, cl1, cl2, p1, p2, m1[j], sig1[j], m2[j], sig2[j], 1000, "MV")
            logging.info(f"Mean error rate for p1={p1} with up to 1000 signals with MV algo for m1={m1[j]}, m2={m2[j]}, sig1={sig1[j]}, sig2={sig2[j]} : {ultimate_mean}")
            plt.savefig(f"figures/simul/Mean_error_rate_MV_p{p1}_{j+1}.png")
            plt.close()


if __name__ == "__main__":
    main()

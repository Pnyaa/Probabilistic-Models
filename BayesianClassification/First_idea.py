#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import multiprocessing
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def configure_logging():
    logging.basicConfig(style='{', format="{asctime} : {message}", datefmt="%c", level=logging.INFO)


def bruit_gauss2(X, cl1, cl2, m1, sig1, m2, sig2):
    bruit1 = np.random.normal(m1, sig1, X.shape)
    bruit2 = np.random.normal(m2, sig2, X.shape)
    return (X == cl1) * bruit1 + (X == cl2) * bruit2


def classif_gauss2(Y, cl1, cl2, m1, sig1, m2, sig2):
    return np.where((norm.pdf(Y, m1, sig1)) > (norm.pdf(Y, m2, sig2)), cl1, cl2)


def taux_erreur(A, B):
    errors = A[A != B]
    return errors.size / A.size


def taux_erreur_moyen(X, cl1, cl2, m1, sig1, m2, sig2, nombre_signaux):
    concat_X = np.tile(X, (1, nombre_signaux)).T
    Y = bruit_gauss2(concat_X, cl1, cl2, m1, sig1, m2, sig2)
    S = classif_gauss2(Y, cl1, cl2, m1, sig1, m2, sig2)
    return taux_erreur(concat_X, S)


def plot_XYS(X, cl1, cl2, m1, sig1, m2, sig2):
    Y = bruit_gauss2(X, cl1, cl2, m1, sig1, m2, sig2)
    S = classif_gauss2(Y, cl1, cl2, m1, sig1, m2, sig2)
    plt.plot(X, label="Signal original")
    plt.plot(S, label="Signal segmenté", linestyle='dashed')
    plt.plot(Y, label="Signal bruité", color="limegreen")
    plt.xlabel("# point d'observation")
    plt.ylabel("Valeur signal")
    # plt.title(f"Reconstruction du signal pour m1={m1}, m2={m2}, sig1={sig1} et sig2={sig2}")
    plt.legend(loc="upper right")
    plt.draw()
    plt.pause(1)


def mean_rate(X, cl1, cl2, m1, sig1, m2, sig2, iterations):
    params = [(X, cl1, cl2, m1, sig1, m2, sig2, i) for i in range(1, iterations + 1)]
    with multiprocessing.Pool() as p:
        mean_rates = p.starmap(taux_erreur_moyen, params)
    ultimate_mean = sum(mean_rates) / len(mean_rates)
    x = np.arange(0, len(mean_rates))
    plt.figure(2, figsize=(8, 6), dpi=80)
    plt.plot(x, mean_rates)
    plt.plot(x, ultimate_mean * np.ones(iterations), color="red", label=f"Erreur moyenne : {ultimate_mean:.3e}")
    # plt.title(f"Evolution du taux d'erreur moyen pour m1={m1}, m2={m2}, sig1={sig1} et sig2={sig2}")
    plt.xlabel("Nombre de signaux sur lesquels l'erreur est moyennée")
    plt.ylabel("Taux d'erreur moyen")
    plt.legend(loc="upper right")
    plt.draw()
    plt.pause(1)
    return ultimate_mean


def main():
    configure_logging()
    signaux = ["signal", "signal1", "signal2", "signal3", "signal4", "signal5"]
    iterations = [5000, 1000, 500, 500, 500, 500]
    for signal in range(len(signaux)):
        X = np.load(f"./signaux/{signaux[signal]}.npy")
        X = X.reshape((X.shape[0], 1))
        cl = np.unique(X)
        cl1, cl2 = cl[0], cl[1]
        m1, m2 = [120, 127, 127, 127, 127], [130, 127, 128, 128, 128]
        sig1, sig2 = [1, 1, 1, 0.1, 2], [2, 5, 1, 0.1, 3]
        t = time.time()
        for i in range(len(m1)):
            plt.figure(1)
            plot_XYS(X, cl1, cl2, m1[i], sig1[i], m2[i], sig2[i])
            plt.savefig(f"figures/Premiere_idee/{signaux[signal]}/XYS_{i+1}.png", bbox_inches="tight")
            plt.close()
            ultimate_mean = mean_rate(X, cl1, cl2, m1[i], sig1[i], m2[i], sig2[i], iterations[signal])
            logging.info(f"Mean error rate for {signaux[signal]}.npy with up to {iterations[signal]} signals for m1={m1[i]}, m2={m2[i]}, sig1={sig1[i]}, sig2={sig2[i]} : {ultimate_mean}")
            plt.savefig(f"figures/Premiere_idee/{signaux[signal]}/Mean_error_rate_{i+1}.png", bbox_inches="tight")
            plt.close()
        logging.info(f"Time taken for {signaux[signal]}.npy : {time.time() - t}")


if __name__ == "__main__":
    main()

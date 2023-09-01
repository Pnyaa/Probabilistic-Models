#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import math
import multiprocessing
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from utils import line_transform_img
from sklearn.cluster import KMeans


def configure_logging():
    logging.basicConfig(style='{', format="{asctime} : {message}", datefmt="%c", level=logging.INFO)


def display(img, pause=1.5, cmap="gray", name=""):
    plt.figure(1)
    plt.imshow(img, cmap=cmap)
    plt.draw()
    plt.axis("off")
    plt.pause(pause)
    if name:
        plt.savefig(f"figures/{name}", bbox_inches="tight")
    plt.close()


def bruit_gauss2(X, cl1, cl2, m1, sig1, m2, sig2):
    return (X == cl1) * np.random.normal(m1, sig1, X.shape) + (X == cl2) * np.random.normal(m2, sig2, X.shape)


def calc_probaprio2(X, cl1, cl2):
    p1 = X[X == cl1].size / X.size
    p2 = X[X == cl2].size / X.size
    return [p1, p2]


def init_EM(Y):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(Y)
    K = kmeans.labels_
    [p1, _] = calc_probaprio2(K, 0, 1)
    m1, m2 = np.mean(Y[K == 0]), np.mean(Y[K == 1])
    sig1, sig2 = np.std(Y[K == 0]), np.std(Y[K == 1])
    return p1, m1, m2, sig1, sig2


def estim_param_EM_indep(iterations, Y, p1, m1, m2, sig1, sig2):
    for _ in range(iterations):
        f1 = norm.pdf(Y, loc=m1, scale=sig1)
        f2 = norm.pdf(Y, loc=m2, scale=sig2)
        pcond_1 = (p1 * f1) / (p1 * f1 + (1 - p1) * f2)
        pcond_2 = (1 - p1) * f2 / (p1 * f1 + (1 - p1) * f2)
        S1 = np.sum(pcond_1)
        S2 = np.sum(pcond_2)
        p1 = (1 / Y.size) * S1
        m1 = np.sum(Y * pcond_1) / S1
        m2 = np.sum(Y * pcond_2) / S2
        sig1 = math.sqrt(np.sum(np.power(Y - m1, 2) * pcond_1) / S1)
        sig2 = math.sqrt(np.sum(np.power(Y - m2, 2) * pcond_2) / S2)
    return p1, 1 - p1, m1, sig1, m2, sig2


def MAP_MPM2(Y, cl1, cl2, p1, p2, m1, sig1, m2, sig2):
    return np.where((p1 * norm.pdf(Y, m1, sig1)) > (p2 * norm.pdf(Y, m2, sig2)), cl1, cl2)


def taux_erreur(A, B):
    return np.count_nonzero(A != B) / A.size


def taux_erreur_moyen(X, cl1, cl2, m1, sig1, m2, sig2, nombre_signaux):
    taux_moyen = []
    for _ in range(nombre_signaux):
        Y = bruit_gauss2(X, cl1, cl2, m1, sig1, m2, sig2)
        flat_Y = line_transform_img(Y).reshape(-1, 1)
        # Initialisation des paramètres EM via K-means
        p1_0, m1_0, m2_0, sig1_0, sig2_0 = init_EM(flat_Y)
        em_p1, em_p2, em_m1, em_sig1, em_m2, em_sig2 = estim_param_EM_indep(100, flat_Y, p1_0, m1_0, m2_0, sig1_0, sig2_0)
        S = MAP_MPM2(Y, cl1, cl2, em_p1, em_p2, em_m1, em_sig1, em_m2, em_sig2)
        taux = taux_erreur(X, S)
        if taux > 0.5:
            taux = 1 - taux
        taux_moyen.append(taux)
    return np.mean(taux_moyen)


def esperance_taux_erreur(X, cl1, cl2, m1, sig1, m2, sig2, iterations):
    params = [(X, cl1, cl2, m1, sig1, m2, sig2, i) for i in range(1, iterations + 1)]
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
    plt.pause(2)
    return ultimate_mean


def save_images(img):
    logging.info(img)
    X = cv2.imread(f"./images/{img}.bmp", cv2.IMREAD_GRAYSCALE)
    display(X)
    m1, m2 = [0, 1, 0], [3, 1, 1]
    sig1, sig2 = [1, 1, 1], [2, 5, 1]
    cl = np.unique(X)
    cl1, cl2 = cl[0], cl[1]
    for i in range(len(m1)):
        Y = bruit_gauss2(X, cl1, cl2, m1[i], sig1[i], m2[i], sig2[i])
        display(Y, name=f"couples/{img}_bruite_{i + 1}")
        flat_Y = line_transform_img(Y).reshape(-1, 1)
        # Initialisation des paramètres EM via K-means
        p1_0, m1_0, m2_0, sig1_0, sig2_0 = init_EM(flat_Y)

        # Initialisation random des paramètres EM
        # random.seed(11)
        # p1_0, m1_0, m2_0, sig1_0, sig2_0 = random.random(), random.random(), random.random(), random.random(), random.random()

        em_p1, em_p2, em_m1, em_sig1, em_m2, em_sig2 = estim_param_EM_indep(100, flat_Y, p1_0, m1_0, m2_0, sig1_0, sig2_0)
        logging.debug(f"Estimated parameters : p1={em_p1:.4f}, p2={em_p2:.4f}, m1={em_m1:.4f}, m2={em_m2:.4f}, sig1={em_sig1:.4f}, sig2={em_sig2:.4f}")
        S = MAP_MPM2(Y, cl1, cl2, em_p1, em_p2, em_m1, em_sig1, em_m2, em_sig2)
        taux = taux_erreur(X, S)
        # Inverse les couleurs attribuées aux classes si erreur > 0.5
        # pour compenser le mélange des classes opéré par k-means
        if taux > 0.5:
            taux = 1 - taux
            S = 255 - S
        logging.info(f"Taux d'erreur pour l'image {img} avec bruit {i + 1} : {taux}")
        display(S, name=f"couples/{img}_segmente_{i + 1}")


def main():
    # Bug avec multiprocessing : https://pythonspeed.com/articles/python-multiprocessing/
    multiprocessing.set_start_method("spawn")
    configure_logging()
    images = ["cible2", "veau2", "zebre2", "alfa2", "country2", "city2", "beee2", "promenade2"]
    for img in images:
        # Génération des images
        save_images(img)

        # Moyenne des taux d'erreur
        # logging.info(img)
        # X = cv2.imread(f"./images/{img}.bmp", cv2.IMREAD_GRAYSCALE)
        # display(X)
        # m1, m2 = [0, 1, 0], [3, 1, 1]
        # sig1, sig2 = [1, 1, 1], [2, 5, 1]
        # cl = np.unique(X)
        # cl1, cl2 = cl[0], cl[1]
        # for i in range(len(m1)):
        #     # Calcul du taux d'erreur moyen
        #     taux_moyen = esperance_taux_erreur(X, cl1, cl2, m1[i], sig1[i], m2[i], sig2[i], 40)
        #     logging.info(f"Taux d'erreur moyen pour l'image {img} bruitée avec le bruit {i + 1} : {taux_moyen}")
        #     plt.savefig(f"figures/couples/erreurs/{img}_bruit{i+1}", bbox_inches="tight")
        #     plt.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import cv2
import random
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from utils import peano_transform_img, transform_peano_in_img
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
    # Attention de bien flatten X avec peano avant d'utiliser cette fonction
    p10 = np.sum(X == cl1) / X.size
    p20 = np.sum(X == cl2) / X.size
    cnt11, cnt12, cnt21, cnt22 = 0, 0, 0, 0
    for i in range(len(X) - 1):
        if X[i] == cl1 and X[i + 1] == cl1:
            cnt11 += 1
        elif X[i] == cl1 and X[i + 1] == cl2:
            cnt12 += 1
        elif X[i] == cl2 and X[i + 1] == cl1:
            cnt21 += 1
        else:
            cnt22 += 1
    A = np.array([[cnt11 / (cnt11 + cnt12), cnt12 / (cnt11 + cnt12)], [cnt21 / (cnt21 + cnt22), cnt22 / (cnt21 + cnt22)]])
    return p10, p20, A


def gauss2(Y, m1, sig1, m2, sig2):
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    return np.concatenate((norm.pdf(Y, m1, sig1), norm.pdf(Y, m2, sig2)), axis=1)


def forward2(Mat_f, A, p10, p20):
    alpha_cl1 = []
    alpha_cl2 = []
    for i in range(Mat_f.shape[0]):
        if i == 0:
            alpha1 = p10 * Mat_f[i, 0]
            alpha2 = p20 * Mat_f[i, 1]
        else:
            alpha1 = (alpha_cl1[-1] * A[0, 0] + alpha_cl2[-1] * A[0, 1]) * Mat_f[i, 0]
            alpha2 = (alpha_cl1[-1] * A[1, 0] + alpha_cl2[-1] * A[1, 1]) * Mat_f[i, 1]
        alpha_cl1.append(alpha1 / (alpha1 + alpha2))
        alpha_cl2.append(alpha2 / (alpha1 + alpha2))
    alpha_cl1 = np.array(alpha_cl1).reshape(-1, 1)
    alpha_cl2 = np.array(alpha_cl2).reshape(-1, 1)
    return np.concatenate((alpha_cl1, alpha_cl2), axis=1)


def backward2(Mat_f, A):
    beta_cl1 = [1 / 2]
    beta_cl2 = [1 / 2]
    # i = 0 déjà initialisé à 1/2, donc boucle jusqu'à len(Mat_f) -1
    for i in range(Mat_f.shape[0] - 1):
        beta1 = A[0, 0] * beta_cl1[0] * Mat_f[-(i + 1), 0] + A[0, 1] * beta_cl2[0] * Mat_f[- (i + 1), 1]
        beta2 = A[1, 0] * beta_cl1[0] * Mat_f[-(i + 1), 0] + A[1, 1] * beta_cl2[0] * Mat_f[- (i + 1), 1]
        beta_cl1.insert(0, beta1 / (beta1 + beta2))
        beta_cl2.insert(0, beta2 / (beta1 + beta2))
    beta_cl1 = np.array(beta_cl1).reshape(-1, 1)
    beta_cl2 = np.array(beta_cl2).reshape(-1, 1)
    return np.concatenate((beta_cl1, beta_cl2), axis=1)


def MPM_chaines2(Mat_f, cl1, cl2, A, p10, p20):
    S = forward2(Mat_f, A, p10, p20) * backward2(Mat_f, A)
    S = S / np.sum(S, axis=1).reshape(-1, 1)
    return np.where(S[:, 0] > S[:, 1], cl1, cl2)


def init_EM(Y):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(Y)
    K = kmeans.labels_
    p10, p20, A = calc_probaprio2(K.reshape(-1, 1), 0, 1)
    m1, m2 = np.mean(Y[K == 0]), np.mean(Y[K == 1])
    sig1, sig2 = np.std(Y[K == 0]), np.std(Y[K == 1])
    return p10, p20, A, m1, m2, sig1, sig2


def estim_param_EM_mc(iterations, Y, A, p10, p20, m1, sig1, m2, sig2):
    N = Y.shape[0]
    for _ in range(iterations):
        Mat_f = gauss2(Y, m1, sig1, m2, sig2)
        alpha = forward2(Mat_f, A, p10, p20)
        beta = backward2(Mat_f, A)
        # On retrouve la probabilité calculée dans MPM_chaine2
        xi = alpha * beta / np.sum(alpha * beta, axis=1).reshape(-1, 1)
        psi11 = alpha[:-1, 0] * A[0, 0] * Mat_f[1:, 0] * beta[1:, 0]
        psi12 = alpha[:-1, 0] * A[0, 1] * Mat_f[1:, 1] * beta[1:, 1]
        psi21 = alpha[:-1, 1] * A[1, 0] * Mat_f[1:, 0] * beta[1:, 0]
        psi22 = alpha[:-1, 1] * A[1, 1] * Mat_f[1:, 1] * beta[1:, 1]
        S = psi11 + psi12 + psi21 + psi22
        psi = np.column_stack((np.column_stack((psi11 / S, psi12 / S)), np.column_stack((psi21 / S, psi22 / S))))
        p10, p20 = np.sum(xi, axis=0) / N
        m1 = (Y.T @ xi[:, 0]) / (N * p10)
        m2 = (Y.T @ xi[:, 1]) / (N * p20)
        sig1 = np.sqrt(np.power((Y - m1), 2).T @ xi[:, 0] / (N * p10))
        sig2 = np.sqrt(np.power((Y - m2), 2).T @ xi[:, 1] / (N * p20))
        A = np.sum(psi, axis=0).reshape(2, 2)
        A[0, :] = A[0, :] / (N * p10)
        A[1, :] = A[1, :] / (N * p20)
    return A, p10, p20, m1[0], sig1[0], m2[0], sig2[0]


def taux_erreur(A, B):
    return A[A != B].size / A.size


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
        display(Y, name=f"Markov/{img}_bruite_{i + 1}")
        pe_Y = peano_transform_img(Y).reshape(-1, 1)
        # Initialisation des paramètres avec k-means
        p10_0, p20_0, A_0, m1_0, m2_0, sig1_0, sig2_0 = init_EM(pe_Y)

        # Initialisation random des paramètres
        # seed = 76
        # random.seed(seed)
        # np.random.seed(seed)
        # p10_0 = random.random()
        # p20_0 =  1 - p10_0
        # m1_0, m2_0, sig1_0, sig2_0 = random.random(), random.random(), random.random(), random.random()
        # A_0 = np.random.rand(2, 2)
        # A_0[0, 1] = 1 - A_0[0, 0]
        # A_0[1, 0] = 1 - A_0[1, 1]

        A, em_p10, em_p20, em_m1, em_sig1, em_m2, em_sig2 = estim_param_EM_mc(50, pe_Y, A_0, p10_0, p20_0, m1_0, sig1_0, m2_0, sig2_0)
        Mat_f = gauss2(pe_Y, em_m1, em_sig1, em_m2, em_sig2)
        logging.debug(f"Estimated parameters : p1={em_p10:.4f}, p2={em_p20:.4f}, m1={em_m1:.4f}, m2={em_m2:.4f}, sig1={em_sig1:.4f}, sig2={em_sig2:.4f}")
        S = MPM_chaines2(Mat_f, cl1, cl2, A, em_p10, em_p20)
        taux = taux_erreur(peano_transform_img(X), S)
        # Inverse les couleurs attribuées aux classes si erreur > 0.5
        # pour compenser le mélange des classes opéré par k-means
        if taux > 0.5:
            taux = 1 - taux
            S = 255 - S
        logging.info(taux)
        display(transform_peano_in_img(S, 256), name=f"Markov/random_init/{img}_segmente_{i + 1}")


def taux_erreur_moyen(X, cl1, cl2, m1, sig1, m2, sig2, nombre_signaux):
    taux_moyen = []
    for _ in range(nombre_signaux):
        Y = bruit_gauss2(X, cl1, cl2, m1, sig1, m2, sig2)
        pe_Y = peano_transform_img(Y).reshape(-1, 1)
        p10_0, p20_0, A_0, m1_0, m2_0, sig1_0, sig2_0 = init_EM(pe_Y)
        A, em_p10, em_p20, em_m1, em_sig1, em_m2, em_sig2 = estim_param_EM_mc(40, pe_Y, A_0, p10_0, p20_0, m1_0, sig1_0, m2_0, sig2_0)
        Mat_f = gauss2(pe_Y, em_m1, em_sig1, em_m2, em_sig2)
        S = MPM_chaines2(Mat_f, cl1, cl2, A, em_p10, em_p20)
        taux = taux_erreur(peano_transform_img(X), S)
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
    plt.pause(1)
    return ultimate_mean


def main():
    # Bug avec multiprocessing : https://pythonspeed.com/articles/python-multiprocessing/
    multiprocessing.set_start_method("spawn")
    configure_logging()
    images = ["cible2", "veau2", "zebre2", "alfa2", "country2", "city2", "beee2", "promenade2"]
    for img in images:
        # Génération d'images
        save_images(img)

        # Calcul taux d'erreur moyen
        # logging.info(img)
        # X = cv2.imread(f"./images/{img}.bmp", cv2.IMREAD_GRAYSCALE)
        # display(X)
        # m1, m2 = [0, 1, 0], [3, 1, 1]
        # sig1, sig2 = [1, 1, 1], [2, 5, 1]
        # cl = np.unique(X)
        # cl1, cl2 = cl[0], cl[1]
        # for i in range(len(m1)):
        #     taux_moyen = esperance_taux_erreur(X, cl1, cl2, m1[i], sig1[i], m2[i], sig2[i], 20)
        #     logging.info(f"Taux d'erreur moyen pour l'image {img} bruitée avec le bruit {i + 1} : {taux_moyen}")
        #     plt.savefig(f"figures/Markov/erreurs/{img}_bruit{i + 1}", bbox_inches="tight")
        #     plt.close()


if __name__ == "__main__":
    main()

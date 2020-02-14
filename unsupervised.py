# -*- coding: utf-8 -*-

''' RIASC Automated Decision Support Software (RADSSo) generates the best supervised/unsupervised model,
    in an automated process, based on some input features and one target feature, to solve a multi-CASH problem.

    Copyright (C) 2018  by RIASC Universidad de Leon (Ángel Luis Muñoz Castañeda, Mario Fernández Rodríguez, Noemí De Castro García y Miguel Carriegos Vieira)
    This file is part of RIASC Automated Decision Support Software (RADSSo)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

	You can find more information about the project at https://github.com/amunc/RADSSo'''

import os
import numpy as np
from sklearn.externals import joblib
from sklearn.cluster import KMeans

import auxiliary_functions as auxf
import global_definitions as glod


def compare_major_classes(diccionario_clases):
    '''Compare the classes to get the major one
    Parameters:
        diccionario_classes: dictionary with classes and the number of values in each one
    return key: cluster number of the majoritary class
    '''
    mayoritaria = glod.get_empty_string()
    key = float(glod.get_nan_string())
    for key in diccionario_clases:
        if mayoritaria == glod.get_empty_string():
            mayoritaria = key
        elif diccionario_clases[key] > diccionario_clases[mayoritaria]:
            mayoritaria = key
    return key


def create_customized_kmeans(train_data, features, target, num_clusters, numero_inicializaciones):
    '''This function allows to create KMeans model using sklearn library
    Parameters:
        train_data: training data
        features: list of relevant features to train the model
        target: objective feature
        num_clusters: number of clusters
        numero_inicializaciones: number of initializations fo the centroids
    return kmeans: Kmeans trained model'''

    init_method = 'k-means++'
    X = train_data[features]
    y = train_data[target]
    if(num_clusters == glod.get_empty_string() and numero_inicializaciones == glod.get_empty_string()):
        kmeans = KMeans(init=init_method)
    else:
        kmeans = KMeans(n_clusters=num_clusters, init=init_method, n_init=numero_inicializaciones)
    kmeans = kmeans.fit(X, y)
    return kmeans


def get_dictionary_of_reasignation_of_labels(modelo, dataset, target):
    '''This function resturns the dictionary with the original labels
    mapped to the ones created by the Kmeans model
    Parameters
    modelo: Kmeans model
    dataset: current data
    target: objective target

    return asociacion_cluster_target: dictionary with original labels mapped to clusters
    '''

    diccionario_plantilla = {}
    clusters = set(list(modelo.labels_))
    relacion_cluster_clase = {}
    clases_originales = list(set(dataset[target]))
    diccionario_clases_originales_contador = {}
    for clase_original in clases_originales:
        diccionario_clases_originales_contador[clase_original] = 0

    for cluster in clusters:
        diccionario_plantilla[cluster] = diccionario_clases_originales_contador.copy()

        indices_cluster = list(np.where(modelo.labels_ == cluster)[0])

        diccionario_contador = {}
        relacion_cluster_clase[cluster] = []
        acum = 0
        for indice in indices_cluster:
            clase = dataset.iloc[int(indice)][target]

            if clase not in diccionario_contador:
                diccionario_contador[clase] = 1
                acum += 1
            else:
                valores = diccionario_contador[clase]
                valores += 1
                acum += 1
                diccionario_contador[clase] = valores

        for clase_original in diccionario_contador:
            diccionario_plantilla[cluster][clase_original] = diccionario_contador[clase_original]

    targets_originales_ordenados = []
    for indice in range(len(clases_originales)):
        for cluster in diccionario_plantilla:
            mayoritario = glod.get_empty_string()
            numero_mayoritario = 0
            for target_original in diccionario_plantilla[cluster]:
                if target_original not in targets_originales_ordenados:
                    if mayoritario == glod.get_empty_string():
                        mayoritario = target_original
                        numero_mayoritario = diccionario_plantilla[cluster][mayoritario]
                    else:
                        numero_candidato = diccionario_plantilla[cluster][target_original]
                        numero_mayoritario, mayoritario = check_majority(numero_candidato, numero_mayoritario, mayoritario, target_original)
            if mayoritario != glod.get_empty_string():
                targets_originales_ordenados.append(mayoritario)

    clases_originales = targets_originales_ordenados

    asociacion_cluster_target = {}
    for clase in clases_originales:
        elementos_cluster_clase = {}
        for cluster_actual in clusters:
            elementos_cluster_clase[cluster_actual] = diccionario_plantilla[cluster_actual][clase]

        mayoritaria = glod.get_empty_string()
        for cluster in elementos_cluster_clase:
            if mayoritaria == glod.get_empty_string():
                mayoritaria = cluster
            else: #comprobamos
                actual = elementos_cluster_clase[mayoritaria]
                candidata = elementos_cluster_clase[cluster]
                if candidata > actual:
                    mayoritaria = cluster

        if mayoritaria != glod.get_empty_string():
            clusters.remove(mayoritaria)
            asociacion_cluster_target[mayoritaria] = clase

    diccionario_plantilla_recodificado = {}
    for cluster in asociacion_cluster_target:
        reco = asociacion_cluster_target[cluster]
        diccionario_plantilla_recodificado[reco] = diccionario_plantilla[cluster]

    return asociacion_cluster_target


def check_majority(numero_candidato, numero_mayoritario, mayoritario, target_original):
    '''This function check if candidate number is greater than majority number'''
    if numero_candidato > numero_mayoritario:
        numero_mayoritario = numero_candidato
        mayoritario = target_original
    return numero_mayoritario, mayoritario

def recodify_list_predicted_targets(lista_predicciones, diccionario_reco):
    '''Recodification of predicted targets using dictionary
    Parameters:
        list lita_predicciones: list wih the predicted values by the unsupervised model
        dict diccionario_reco: dictionary that maps clusters - original targets
    return predicciones_reco: predictions labelled according to original targets'''

    predicciones_reco = []
    for elemento in lista_predicciones:
        reco = diccionario_reco[elemento]
        predicciones_reco.append(reco)
    return predicciones_reco


def get_accuracy(lista_targets_datos_catalogados, targets_predichos_recodificados):
    '''It allows to compute accuracy for unsupervised model
    Parameters:
       list lista_targets_datos_catalogados:  list with original values for the target
       list targets_predichos_recodificados: list with predicted values for the target
    return:
        int accuracy: computed accuracy
    '''
    accuracy = auxf.compare_lists(targets_predichos_recodificados, lista_targets_datos_catalogados)/float(len(lista_targets_datos_catalogados))
    return accuracy

def save_model_to_disk(modelo, ruta_directorio, nombre_fichero):
    '''
    It allows to create a .pkl wit the scikit-learn model trained

    Parameters:
    :param sklearn-model model: trained model using scikit-learn
    :param str ruta_directorio: full path to the directory that contains the saved model
    :param str nombre_fichero: filename for the file that stores the model (ext. must be included)

    :return: None
    '''

    ruta_destino = os.path.join(ruta_directorio, nombre_fichero)
    joblib.dump(modelo, ruta_destino)
    return ruta_destino

def initialize_model(model_name, params_array, diccionario_modelos_no_supervisado):
    '''
    The function allows to create and train a model with the specified name

    Parameters:
    :param str model_name: name of the model to be trained
    :param pandas-dataframe train_data: Data to train de model. It includes the target column
    :param list features: List with the relevant features to rain the model
    :param str target: Target feature
    :param list params_array: Array with the specific parameters of the model to be trained

    :return: specified model trained
    :rtype: sklearn-model

    '''

    modelo_inicializado = glod.get_empty_string()
    if model_name == diccionario_modelos_no_supervisado[1]:#Kmeans
        modelo_inicializado = KMeans(n_clusters=params_array[0], n_init=params_array[1])

    return modelo_inicializado

def create_trained_model(model_name,
                         train_data,
                         features_target,
                         params_array,
                         diccionario_modelos_no_supervisado):
    '''
    The function allows to create and train a model with the specified name

    Parameters:
    :param str model_name: name of the model to be trained
    :param pandas-dataframe train_data: Data to train de model. It includes the target column
    :param list features: List with the relevant features to rain the model
    :param str target: Target feature
    :param list params_array: Array with the specific parameters of the model to be trained

    :return: specified model trained
    :rtype: sklearn-model
    '''
    features = features_target[0]
    target = features_target[1]
    modelo_creado = glod.get_empty_string()
    if model_name == diccionario_modelos_no_supervisado[1]:#Kmeans
        modelo_creado = create_customized_kmeans(train_data,
                                                 features,
                                                 target,
                                                 params_array[0],
                                                 params_array[1])

    return modelo_creado

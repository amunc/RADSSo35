# -*- coding: utf-8 -*-

''' RIASC Automated Decision Support Software (RADSSo) 2.0 generates the best supervised/unsupervised model,
    in an automated process, based on some input features and one target feature, to solve a multi-CASH problem.

    Copyright (C) 2018  by RIASC Universidad de Leon (Ángel Luis Muñoz Castañeda, Mario Fernández Rodríguez, Noemí De Castro García y Miguel Carriegos Vieira)
    This file is part of RIASC Automated Decision Support Software (RADSSo) 2.0

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

	You can find more information about the project at https://github.com/amunc/RADSSo35'''

import os
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import auxiliary_functions as auxf
import global_definitions as glod

def calculate_model_accuracy(model, target_data_set, features, target, out):
    '''
    This function allows to obtain the percentage of success of the model over
    the target_dataset received as parameter

    Parameters:
    :param sklearn-model model: scikit-learn learning model
    :param pandas-dataframe target_data_set: dataset with all the observations
        and relevant features (must include objective target feature).
    :param list features: list with relevant features used to train the model
    :param str target: objective feature for which the model was trained
    :param int out: Detailed information about accuracy is shown when it is set to 1.

    :return: percentage of success
    :rtype: float
    '''

    predictions = model.predict(target_data_set[features])
    predictions = list(predictions)
    original_targets = list(target_data_set[target])
    accuracy = auxf.compare_lists(predictions, original_targets)/float(len(original_targets))
    accuracy = float(round(accuracy, 5))
    if out == 1:
        print('accuracy=', accuracy)
    return accuracy


def save_current_model_to_file(modelo, ruta_directorio, nombre_fichero):
    '''
    This function allows to get a .pkl with the created scikit-learn model

    Parameters:
    :param sklearn-model model: supervised scikit-learn model
    :param str ruta_directorio: full path to the directory that stores the model
    :param str nombre_fichero: filename with extension of the file that will
        contain the model

    :return: None
    '''

    ruta_destino = os.path.join(ruta_directorio, nombre_fichero)
    joblib.dump(modelo, ruta_destino)
    return ruta_destino


def create_customized_tree(train_data, features, target, depth):
    '''
    This function allows to create the DecisionTreeClassifier model using sckikit-learn

    Parameters:
    :param pandas-dataframe train_data: Training data that include target feature
    :param list features: List with relevant features to train the model
    :param str target: Objtective feature for the model
    :param int depth: Desired depth for the trees that constitute the model

    :return: Trained DecisionTreeClassifier model
    :rtype: sklearn-model
    '''

    criterion_used = "entropy"
    X = train_data[features]
    y = train_data[target]
    if depth == glod.get_empty_string():
        decision_tree = DecisionTreeClassifier(criterion=criterion_used)
    else:
        decision_tree = DecisionTreeClassifier(max_depth=int(depth), criterion=criterion_used)
    my_tree = decision_tree.fit(X, y)
    return my_tree


##################################################################################################
################################## MODELOS ENSEMBLE ##############################################
##################################################################################################

def create_customized_ada(train_data, features, target, estimators):
    '''
    It allows to create AdaBoostClassifier model using scikit-learn

    Parameters:
    :param pandas-dataframe train_data: Training data that include target feature
    :param list features: List with relevant features to train the model
    :param str target: Objtective feature for the model
    :param int estimators: Number of estimators that compose the model

    :return: Trained AdaBoostClassifier model
    :rtype: sklearn-model
    '''

    X = train_data[features]
    y = train_data[target]
    if estimators == glod.get_empty_string():
        mi_ada = AdaBoostClassifier(base_estimator=None,
                                    learning_rate=1.0,
                                    algorithm='SAMME.R',
                                    random_state=None)
    else:
        mi_ada = AdaBoostClassifier(base_estimator=None,
                                    n_estimators=estimators,
                                    learning_rate=1.0,
                                    algorithm='SAMME.R',
                                    random_state=None)
    ada = mi_ada.fit(X, y)
    return ada


def create_customized_boosting(train_data, features, target, depth, estimators):
    '''
    This function allows to create the GradientBoostingClassifier usning scikit-learn

    Parameters:
    :param pandas-dataframe train_data: Training data that include target feature
    :param list features: List with relevant features to train the model
    :param str target: Objtective feature for the model
    :param int estimators: Number of estimators that compose the model
    :param int depth: Desired depth for the models that constitute the model

    :return: Trained GradientBoostingClassifier model
    :rtype: sklearn-model
    '''

    X = train_data[features]
    y = train_data[target]
    if(estimators == glod.get_empty_string() and depth == glod.get_empty_string()):
        gradientboosting = GradientBoostingClassifier()
    else:
        gradientboosting = GradientBoostingClassifier(n_estimators=estimators, max_depth=depth)
    boosting = gradientboosting.fit(X, y)
    return boosting


def create_customized_forest(train_data, features, target, depth, estimators):
    '''
    This function allows to create RandomForestClassifier using scikit-learn

    Parameters:
    :param pandas-dataframe train_data: Training data that include target feature
    :param list features: List with relevant features to train the model
    :param str target: Objtective feature for the model
    :param int estimators: Number of estimators that compose the model
    :param int depth: Desired depth for the trees that constitute the model

    :return: Trained RandomForestClassifier model
    :rtype: sklearn-model
    '''

    criterion_used = "entropy"
    X = train_data[features]
    y = train_data[target]
    if(estimators == glod.get_empty_string() and depth == glod.get_empty_string()):
        random_forest = RandomForestClassifier(criterion=criterion_used)
    else:
        random_forest = RandomForestClassifier(n_estimators=estimators,
                                               criterion=criterion_used,
                                               max_depth=depth)
    forest = random_forest.fit(X, y)
    return forest


##################################################################################################
########################## MODELOS REDES NEURONALES ##############################################
##################################################################################################

def create_customized_mlp(train_data, features, target, layer_sizes, act_function):
    '''
    The function allows to create MLPClassifier model using scikit-learn

    Parameters:
    :param pandas-dataframe train_data: Training data that include target feature
    :param list features: List with relevant features to train the model
    :param str target: Objtective feature for the model
    :param tuple layer_sizes: Tuple with the number of layers and percentrons in each one
    :param str act_func: activation function for the layer

    :return: Trained MLPClassifier model
    :rtype: sklearn-model
    '''

    solver_used = 'adam'
    X = train_data[features]
    y = train_data[target]
    if(layer_sizes == glod.get_empty_string() and act_function == glod.get_empty_string()):
        mlp = MLPClassifier()
    else:
        mlp = MLPClassifier(hidden_layer_sizes=layer_sizes,
                            activation=act_function,
                            solver=solver_used)
    my_mlp = mlp.fit(X, y)
    return my_mlp



def get_trained_model(model_name, train_data, features_target, params_array,
                      diccionario_modelos_supervisado):
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

    if model_name == diccionario_modelos_supervisado[1]: #Tree
        modelo_creado = create_customized_tree(train_data, features, target, params_array[0])

    elif model_name == diccionario_modelos_supervisado[2]:#Ada
        modelo_creado = create_customized_ada(train_data, features, target, params_array[0])

    elif model_name == diccionario_modelos_supervisado[3]:#Boosting
        modelo_creado = create_customized_boosting(train_data, features, target,
                                                   params_array[0], params_array[1])

    elif model_name == diccionario_modelos_supervisado[4]:#RandomForest
        modelo_creado = create_customized_forest(train_data, features, target,
                                                 params_array[0], params_array[1])

    elif model_name == diccionario_modelos_supervisado[5]:#MLP
        modelo_creado = create_customized_mlp(train_data, features, target,
                                              params_array[0], params_array[1])

    return modelo_creado

def initialize_model(model_name, params_array, diccionario_modelos_supervisado):
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
    criterion_used = "entropy"
    algorithm_used = 'SAMME.R'

    if model_name == diccionario_modelos_supervisado[1]:#Tree
        modelo_inicializado = DecisionTreeClassifier(max_depth=params_array[0], criterion=criterion_used)

    elif model_name == diccionario_modelos_supervisado[2]:#Ada
        modelo_inicializado = AdaBoostClassifier(base_estimator=None,
                                                 n_estimators=params_array[0],
                                                 learning_rate=1.0,
                                                 algorithm=algorithm_used, random_state=None)

    elif model_name == diccionario_modelos_supervisado[3]:#Boosting
        modelo_inicializado = GradientBoostingClassifier(n_estimators=params_array[0],
                                                         max_depth=params_array[1])

    elif model_name == diccionario_modelos_supervisado[4]:#RandomForest
        modelo_inicializado = RandomForestClassifier(n_estimators=params_array[1],
                                                     criterion=criterion_used,
                                                     max_depth=params_array[0])

    elif model_name == diccionario_modelos_supervisado[5]:#MLP
        modelo_inicializado = MLPClassifier(hidden_layer_sizes=(params_array[0]),
                                            activation=params_array[1])

    return modelo_inicializado

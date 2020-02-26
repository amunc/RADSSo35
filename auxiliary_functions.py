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

import codecs
import copy
import datetime
import json
import numpy as np
import os
import pandas as pd
import pickle
import shutil
import simplejson
import time
import io


import data_request as datr
import reports_generation as repg
import global_definitions as glod

def print_initial_license_message():
    print(''' 
    
     RIASC Automated Decision Support Software (RADSSo) 2.0 generates the best supervised/unsupervised model,
    in an automated process, based on some input features and one target feature, to solve a multi-CASH problem.

    Copyright (C) 2018  by RIASC Universidad de Leon (Angel Luis Munoz Castaneda, Mario Fernandez Rodriguez, Noemi De Castro Garcia y Miguel Carriegos Vieira)
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
	
	 You can find more information about the project at https://github.com/amunc/RADSSo\n''')
    
    
def print_debug_info(compute_lc,penalize_falses,default_matrix,validation_mode,exponent,main_metric,feature_selection_method,n_feat,percentil,name_variable_with_events,target,label_non_catalogued,w):
    print("\n--- DEBUG INFO ---")
    print("Compute learning curves: ",compute_lc)
    print("Penalize falses: ",penalize_falses)
    print("Default matrix: ",  default_matrix)
    print("Validation mode: " , validation_mode)
    print("Exponent value: " + str(exponent))
    print("Main metric: ", main_metric)
    print("Feature selection method ", feature_selection_method)
    print("Percentil: ",percentil)
    print("Percentage of features inside percentil: " + str(n_feat))
    print("Events feature: ", name_variable_with_events)
    print("Target feature: ", target)
    print("Weighted matrix: \n", w)    
    print("--- DEBUG INFO ---\n")
    

def load_json_file(fullpath_to_filename):
    '''
    This function allows to retrieve a dictionary in json format stored in
    a json file

    Parameters:    
    :param str fullpath_to_filename: String with the full path to the location of the
    json file
    
    :exception IOError is json file is not found
    
    :return: dictionary: Json parameters in dictionary form
    :rtype: dict
    '''
    
    if(os.path.lexists(fullpath_to_filename)):
        #read_file = open(fullpath_to_filename,"r",encoding="utf-8").read()        
        read_file = io.open(fullpath_to_filename, mode="r", encoding="utf-8").read()
        dictionary = simplejson.loads(read_file)
    else:
        raise IOError('File ' + fullpath_to_filename.encode(glod.get_encoding()) + ' does not exist')
    return dictionary

	
def check_input_target(vector_with_input_arguments):
    '''
    This function allows to check if the number of invocation parameters introduced by command line is correct.

    Parameters:    
    :param list vector_with_input_arguments: List with the arguments introduced by command line
    
    :exception Exception if target is not correct or if invocation is not correct
    
    :return: target,Name of the target
    :rtype: str
    '''
    
    target = ''
    if(len(vector_with_input_arguments) > 1):
        target = vector_with_input_arguments[1]
    else:
        print ('main_train.py invocation is not correct: main_train.py target_feature_name [' +  glod.get_semi_automated_mode_name() + '] [' + glod.get_build_files_mode_name() + '] ')
        raise Exception('main_train.py invocation is not correct: main_train.py target_feature_name [' +  glod.get_semi_automated_mode_name() + '] [' + glod.get_build_files_mode_name() + '] ')
    return target
	

def compute_matrix_weights_by_default(dimensions):
    '''
    This function allows compute the matrix of weights by using
    an uniform distribution for each weight

    Parameters:    
    :param int dimensions: Number of values in the target feature 
    
    :return: np_matrix: Matrix of weights in list form
    :rtype: list
    '''
    
    value_to_fullfil = float(1.0/dimensions)
    np_matrix = np.full((dimensions,dimensions),value_to_fullfil)
    return np_matrix.tolist()


def compute_matrix_weights(dimensions,diagonal_value):
    '''
    This function allows compute the matrix of weights by using
    an uniform distribution for each weight or the value specified
    by the user for the diagonal

    Parameters:    
    :param int dimensions: number of values in the target feature
    :param float digonal_value: float value for the diagonal specified by the user
    
    :exception ValueError if diagonal value is lower than 0.5 or other
    :exception Exception if an unexpected exception occurs while computing the values
    for the matrix
        
    :return: matrix_of_weights: Matrix of weights in list form
    :rtype: list
    ''' 
    
    try:
        diagonal_value  = float(diagonal_value)
        if(diagonal_value<=0.5):
            raise ValueError()
        else:
            value_to_fullfil = float(1.0 - diagonal_value)
            value_to_fullfil = float(value_to_fullfil/(dimensions-1))
            np_matrix = np.full((dimensions,dimensions),value_to_fullfil)
            for indice_diagonal in range(0,dimensions):
                np_matrix[indice_diagonal,indice_diagonal] = diagonal_value
            matrix_of_weights = np_matrix        
    except Exception:
        matrix_of_weights = compute_matrix_weights_by_default(dimensions)
    return matrix_of_weights
    

def check_matrix_weights(matrix_of_weights,dimensions):
    '''
    This function allows to check if the matrix of weights in the conf.ini
    is correct in shape and in values.

    Parameters:    
    :param list matrix_of_weights: Matrix with the weights introduced by the user. Must be a 3x3 matrix which values of rows
    sum a total amount of 1.0
    :param int dimensions: Right number of dimensions for the matrix.
    
    :exception ValueError if shape ot he matrix or distribution of values is not correct
    
    :return: checked_matrix_of_weights: Matrix with weights already checked
    :rtype: numpy_array
    '''
    
    matrix_of_weights = np.array(matrix_of_weights)    
    checked_matrix_of_weights =[]
    if(matrix_of_weights.shape[0]== dimensions):
        correct_number_elements = dimensions*dimensions
        number_elements = 0
        for indice in range(0,dimensions):
            number_elements+=len(matrix_of_weights[indice])        
        if(number_elements == correct_number_elements):          
            correct_distribution = True
            contador = 0
            while(contador <= dimensions-1 and correct_distribution):
                row_distribution = round(np.sum(matrix_of_weights[contador]),0)                
                if(row_distribution != 1.0):
                    correct_distribution = False
                contador+=1
            if(correct_distribution):
                checked_matrix_of_weights = matrix_of_weights.tolist()
            else:
                raise ValueError('Check conf.ini - matrix_of_weights_fp_fn parameter:\n The distribution in any row of the matrix of weights is not correct (the addition of elements in the rows must be 1.0).')            
        else:
            raise ValueError("Check conf.ini - matrix_of_weights_fp_fn parameter:\n The shape of the weighted matrix must be num_target_values(" + str(dimensions) + ") rows by num_target_values(" + str(dimensions) + ") columns.Number of columns incorrect")
    else:
        raise ValueError("Check conf.ini - matrix_of_weights_fp_fn parameter:\n The shape of the weighted matrix must be num_target_values(" + str(dimensions) + ")rows by num_target_values(" + str(dimensions) + ") columns. Number of rows incorrect")
    
    return checked_matrix_of_weights
	
	
def check_train_test_division_percent(percent):
    '''
    This function allows to check if the division percentage for the train
    and test datasets is correct

    Parameters:    
    :param float percent: pecentage of division introduced by the user
    
    :exception ValueError if the percentage is out of range(0.6,0.9)
    
    :return: checked_percent, percent of division
    :rtype: float
    '''
    
    if(percent < 0.6 or percent > 0.9):
        raise ValueError("Check conf.ini - Parameter with the percentage of division in train and test datasets is not in the range[0.6,0.9]")
    else:
        return percent


def save_dictionary_to_disk_pickle_format(dictionary_to_store,handler):
    '''
    This function allows to store the dict structure that receives as parameter
    closing it with making use of the correct handler.

    Parameters:
    :param dict dictionary_to_store: Dictionary structure to store in disk.
    :param file handler: Handler for the dictionary_to_store    
    
    :return: None    
    '''
    
    pickle_protocol=pickle.HIGHEST_PROTOCOL
    pickle.dump(dictionary_to_store, handler, protocol=pickle_protocol)
    handler.close()


def open_dictionary_pickle_format_for_reading(full_path_to_stored_dictionary):
    '''
    This function allows to create a new dictionary and to open it using the
    path received as parameter

    Parameters:    
    :param str full_path_to_stored_dictionary: String with the full path to the
    location of the dictionary that will be loaded.
    
    :return: retrieved_dictionary: Dictionary stored previously
    :rtype: dict<python_hashable_type:python_type>
    :return: handler: Handler to operate the diectionary
    :rtype: file
    '''
    
    handler = open(full_path_to_stored_dictionary, 'rb')
    retrieved_dictionary = pickle.load(handler)
    return retrieved_dictionary,handler


def get_dictionary_in_pickle_format_handler(fullpath_to_dictionary):
    '''
    This function allows to obtain dictionary handler from the
    the specified path with the specified name in order to modify it

    Parameters:    
    :param str fullpath_to_dictionary: String with the full path to the location of the
    dictionary that will be loaded.
    
    :return: handler, the handler of the dictionary previously stored
    :rtype: file handler
    '''
    
    handler = open(fullpath_to_dictionary, 'wb')
    return handler


def retrieve_dictionary_and_handler(fullpath_to_dictionary):
    '''
    This function allows to retrieve the dictionary stored in disk with the corresponding
    handler (it can be a new one or a dictionary that was stored previously)

    Parameters:    
    :param str fullpath_to_dictionary: String with the full path to the location of the
    dictionary that will be loaded.
    
    :return: returned_dictionary: Dictionary in the fullpath_to_dictionary
    :rtype: dict<python_hashable_type:python_type>
    :return: handler: The handler of the dictionary
    :rtype: file
    '''
    
    returned_dictionary = {}
    handler=None
    if(os.path.exists(fullpath_to_dictionary)):
        returned_dictionary,handler = open_dictionary_pickle_format_for_reading(fullpath_to_dictionary)
    else: 
        pickle_protocol=pickle.HIGHEST_PROTOCOL
        handler =  open(fullpath_to_dictionary, 'wb')
        pickle.dump(returned_dictionary, handler, protocol=pickle_protocol)
        handler.close()        
        returned_dictionary,handler = open_dictionary_pickle_format_for_reading(fullpath_to_dictionary)
    return returned_dictionary,handler


def restore_dictionary_to_previous_version(fullpath_to_dictionary,previous_version_of_dict):
    '''
    This function allows to restore the previous safe version of the dictionary when
    the last modification was being saved and it was interrupted

    Parameters:    
    :param str fullpath_to_dictionary: String with the full path to the location of the
    dictionary that will be loaded.
    :param dict previous_version_of_dict: State of the before the last changes in the dictionary
    
    :return: None    
    '''
    
    pickle_protocol=pickle.HIGHEST_PROTOCOL
    handler =  open(fullpath_to_dictionary, 'wb')
    pickle.dump(previous_version_of_dict, handler, protocol=pickle_protocol)
    handler.close()    


def check_number_of_values_objective_target(number_different_values):
    '''
    This function allows to check the number if targets to determine if the problem
    has a solution applying machine learning.

    Parameters:    
    :param int number_different_values: number of targets found
    
    :return: True | False
    :rtype: boolean
    '''
    
    classification_problem_already_solved = False    
    if(number_different_values < 2):
        classification_problem_already_solved = True
    return classification_problem_already_solved

    
def get_number_different_values_objective_target(list_with_all_values_of_objective_target):
    '''
    This function allows to obtain the number values for the target feature of the current event.

    Parameters:    
    :param list list_with_all_values_of_objective_target: Colum with all the values for the objective feature
    
    :return: The number of values for the target feature
    :rtype: int
    '''
    return len(list(set(list_with_all_values_of_objective_target)))
    
    
def compare_lists(first_list,second_list):
    '''
    This function allows to compare two lists of elements and for each element that is located at the same position in both lists,
    in increases the counter variable number_of_coincident_elements. Both lists must have the same length

    Parameters:
    :param list first_list: First list with the elements to compare.
    :param list second_list: Second list with the elements to compare.  
    
    :return: number of elements that are the same in the same location at both lists
    :rtype: int
    '''
   
    number_of_coincident_elements=0
    for i in range(len(first_list)):
        if (first_list[i] == second_list[i]):            
            number_of_coincident_elements=number_of_coincident_elements+1
    return(number_of_coincident_elements)


def get_complementary_list(full_list,sublist):
    '''
    This function allows to obtain the complementary list of a sublist recevied as parameter
    and that is a sublist of the another list received as parameter from which are extracted the
    complementary features

    Parameters:
    :param list full_list: List with all the elements
    :param list sublist: List with some elements of full list (a sublist of full_list)
    
    :return: complementary_list,number of elements that are only in full_list
    :rtype: list'''
   
    complementary_list=list(set(full_list).difference(sublist))
    return(complementary_list)
    

def get_all_files_in_dir_with_extension(fullpath_to_directory_with_files,maximum_number_files_to_read,files_extension):
    '''
    This function allows to create a list(vector) with all the files in a directory that
    has an specific files_extension
    
    Parameters:
    :param str fullpath_to_directory_with_files: Path to the root directory with the files.
    :param int maximo_numero_ficheros: Number that indicates the maximum number of files that will be loaded.   
    :param files_extension: Extension of the files to look for into the directory

    :return: vector_with_fullpaths_to_input_files,list with strings that refers to the full path of each file to be read orderer by name    
    :rtype: list<str>
    '''
    
    vector_with_fullpaths_to_input_files = []
    for root, directories, available_files in os.walk(fullpath_to_directory_with_files):  
        for filename in available_files:
            if('.'+files_extension in filename):                
                vector_with_fullpaths_to_input_files.append(os.path.join(fullpath_to_directory_with_files,filename))
                
    if (maximum_number_files_to_read !=0 ):
        if(len(vector_with_fullpaths_to_input_files) > maximum_number_files_to_read):
            vector_with_fullpaths_to_input_files = vector_with_fullpaths_to_input_files[0:maximum_number_files_to_read]
    return sorted(vector_with_fullpaths_to_input_files)
    
    
def checking_events_reading_method(vector_input_arguments,vector_of_fullpaths_to_input_files,fullpath_to_json_with_events_features,feature_events_name,input_files_delimiter,execution_log,time_log,list_of_not_valid_characters,enco):
    '''
    This function allows several things depending on the invocation of main.py:
        main.py target (2 arguments): it reads the events and variables from the path fullpath_to_json_with_events_features (a file created by the user is necesary)
        main.py target build_files(3 arguments): it drives the user to build a file with events and variables that will be used during the process
        main.py target auto (3 arguments): it tries to extract the events automatically from the input files
    
    
    Parameters:
    :param list vector_input_arguments: List with the parameters introduced by command line
    :param list vector_of_fullpaths_to_input_files: List with the path to the files that will be read
    :param str fullpath_to_json_with_events_features: Full path to the file with events, relevant features and discarded features
    :param str feature_events_name: Feature inside the input files where the name of the events is stored    
    :param str input_files_delimiter: Field delimiter for input files 
    :param str execution_log: Path to file log to write 
    :param str time_log: Path to file time log to write
    :list_of_not_valid_characters: List of characters that can not be used in a path name
    
    :return list lista_eventos: list of events
    :rtype: list<str>
    :return list_list_of_relevant_features_for_the_user: list of relevant features for each event specified by the user
    :rtype: list<list<str>>
    :return list_list_discarded_features_by_the_user: list of discarded variables for each event specified by the user
    :rtype: list<list<str>>    
    '''
        
    substep_init_time = datetime.datetime.fromtimestamp(time.time())
    list_of_events_to_process = []
    list_list_of_relevant_features_for_the_user = []
    list_list_discarded_features_by_the_user = []
    
    '''Checking if automatic mode has been selected'''    
    if(len(vector_input_arguments) > 2 and vector_input_arguments[2] == glod.get_semi_automated_mode_name()): #Manual reading (building files or using existing ones)
        repg.register_log([execution_log],'>>>>>>Substep 1.3: Recognizing events manually '+ substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n\n" \
                       ,'',enco)
        if(len(vector_input_arguments) > 3 and vector_input_arguments[3] == glod.get_build_files_mode_name()): #Building files option specified            
            repg.register_log([execution_log],'>>>>>>Substep 1.3: Building files by user interaction '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n\n" \
                       ,'',enco)          
            diccionario_eventos_variables = datr.get_events_and_variables_from_user_interaction(vector_of_fullpaths_to_input_files, feature_events_name, input_files_delimiter,list_of_not_valid_characters,enco)            
            with codecs.open(fullpath_to_json_with_events_features, 'w',encoding=enco) as outfile:
                json.dump(diccionario_eventos_variables, outfile,ensure_ascii=False)
                                
        '''Reading json with events and variables'''        
        list_of_events_to_process,list_list_of_relevant_features_for_the_user,list_list_discarded_features_by_the_user = datr.get_list_events_features_json_file(fullpath_to_json_with_events_features,list_of_not_valid_characters)
                                
        substep_finish_time = datetime.datetime.fromtimestamp(time.time())
        repg.register_log([time_log],'>>>>>>Substep 1.3 Recognizing events manually elapsed time: '+ str(substep_finish_time - substep_init_time) + "\n",'',enco)
        repg.register_log([execution_log],'>>>>>>Substep 1.3 (manually) ends '+ substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    else:        
        repg.register_log([execution_log],'>>>>>>Substep 1.3: Recognizing events automatically '+ substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n\n" \
                       ,'',enco)        
        list_of_events_to_process,list_of_features = datr.recognize_available_events_and_variables_automatically(vector_of_fullpaths_to_input_files, feature_events_name, input_files_delimiter, list_of_not_valid_characters,enco)
            
        list_of_events_to_process=sorted(list_of_events_to_process)
        for event in list_of_events_to_process:
            list_list_of_relevant_features_for_the_user.append([])
            list_list_discarded_features_by_the_user.append([])
        substep_finish_time = datetime.datetime.fromtimestamp(time.time())
        repg.register_log([time_log],'>>>>>>Substep 1.3 Recognizing events automatically elpased time: '+ str(substep_finish_time - substep_init_time) + "\n",'',enco)
        repg.register_log([execution_log],'>>>>>>Substep 1.3 (automatically) ends '+ substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
            
    return list_of_events_to_process,list_list_of_relevant_features_for_the_user,list_list_discarded_features_by_the_user


def create_directory(fullpath_to_directory):
    '''
    This function allows to create a non-existent directory in the path specified.If the directory already exists no action is performed.    
    If the directory cannot be created a ValueException is raised.
    
    Parameters:
    :param str fullpath_to_directory: Full path to the location of the new directory that includes the name of the directory to be created.    
    :exception ValueError: If the directory could not be created
    
    :return: None
    '''
    
    try:
        os.makedirs(fullpath_to_directory)
        
    except OSError: 
        pass
    
    except Exception:
        raise ValueError("Directory could no be created " + str(fullpath_to_directory))


def delete_directory_with_content(fullpath_to_directory):
    '''
    This function allows to remove the specified directory
    
    Parameters:
    :param str fullpath_to_directory: Full path to the location of the directory to erase    
    
    :return: None
    '''
    
    shutil.rmtree(fullpath_to_directory,ignore_errors=True)

    
def load_dictionary_of_features(fullpath_to_json_file,json_key):
    '''
    It allows to obtain a dictionary with the variables hardcoded, by this way if the name of any of the variables is modified, 
    it would be only necessary to change the value for that key in the dictionary.
     
    :param str fullpath_to_json_file: Full path to the json where the dictionary with the variables and the names (that they receive
    in the current execution) is stored
    :exception IOError: If the json file is not found
    :exception KeyError: If the dictionray with the variables is not found

    :return: dictionary with features with known recodification and their actual name
    :rtype: dict<str:str>
    '''
    
    json_variables_known_recodification = ''
    json_variables_known_recodification = load_json_file(fullpath_to_json_file)        
        
    return json_variables_known_recodification[json_key]
    


def remove_feature_from_list(feature_to_remove,list_of_features):
    '''
    This function allows to delete the variable, that is passed as parameter, from the list

    Parameters:
    :param str feature_to_remove: variable to delete from the list
    :param list<str> list_of_features: list of variables where to find the variable to delete
    
    :return: list with the non-deleted variables
    :rtype: list<str>
    '''
    
    if feature_to_remove in list_of_features:
        list_of_features.remove(feature_to_remove)    
    return list_of_features


def count_number_of_observations_by_target(df_data,target):
    '''
    The function allows to count the number of observations that have each one of the existent value for the target variable in df_datos

    Parameters:
    :param pandas_dataframe df_datos: dataframe with all data available where the target feature is included
    :param str target: objective variable
    
    :return: dictionary_with_value_for_target_and_occurrences,dictionary where the keys are the differents values of the targets and the values are the number of observations under the particular target
    :rtype: dict<int:int>
    '''
    
    lista_targets = df_data[target]
    dictionary_with_value_for_target_and_occurrences = {}
    for current_target in lista_targets:
        if(current_target not in dictionary_with_value_for_target_and_occurrences):
            dictionary_with_value_for_target_and_occurrences[current_target] = 1
        else:
             valores = dictionary_with_value_for_target_and_occurrences[current_target]   
             valores+=1
             dictionary_with_value_for_target_and_occurrences[current_target] = valores
    
    return dictionary_with_value_for_target_and_occurrences


def deleting_empty_and_constant_features(df_data):
    '''
    This function allows to remove empty and constant features from a dataframe

    Parameters:
    :param pandas_Dataframe df_data: Dataframe with all available data
    
    :return: list_valid_features: sorted list of non-empty and non-constant features
    :rtype: list<str>
    :return: list_of_empty_or_constant_features: sorted list of empty or constant features
    :rtype: list<str>
    '''

    '''Full list of available features to check'''
    list_features_to_check = list(df_data.columns)
    
    '''Looping through the list to find empty or constant features'''
    list_of_empty_or_constant_features = []
    for i in range(len(list_features_to_check)):        
        current_feature = list_features_to_check[i]
        
        list_of_elements = pd.unique(df_data[current_feature])
        list_of_elements = list_of_elements.tolist()
        if len(list_of_elements) == 1:                                    
            list_of_empty_or_constant_features.append(current_feature)
    
    '''Obtaining the list with the variable non-empty and non-constant'''
    list_valid_features= get_complementary_list(list_features_to_check,list_of_empty_or_constant_features)
    
    return sorted(list_valid_features), sorted(list_of_empty_or_constant_features)


def get_features_to_train_models(list_relevant_features_and_scores,features_discarded_scoring_process,percentage):
    '''
    Allows to obtaind the relevant features without those that were specified by the user
    with the possibility of selecting only a pecentage

    Parameters:
    :param list<str> list_relevant_features_and_scores: list of variables and scores from MIF of each one
    :param list<str> features_discarded_scoring_process: lsit of variables specified by the user to be discarded
    
    :return: list_top_features, list with relevant features after scoring function
    :rtype: list<str>
    :return: dictionary_relevant_features_scores, dictionary with relevant features and their corresponding score
    :rtype: list<str>
    '''
    
    dictionary_relevant_features_scores = {}
    list_top_features = []
    num_final = int(len(list_relevant_features_and_scores)*percentage)
    
    for element in list_relevant_features_and_scores[0:num_final+1]:
        dictionary_relevant_features_scores[element[0]] = round(float(element[1]),6)
        list_top_features.append(element[0])
        
    for feature in features_discarded_scoring_process:
        if (feature in dictionary_relevant_features_scores):
            list_top_features.remove(feature)
            del dictionary_relevant_features_scores[feature]
                        
    return list_top_features,dictionary_relevant_features_scores

	
def split_dataset_catalogued_non_catalogued(vector_with_fullpath_to_input_files,feature_events_name,event,target,maximum_number_of_registers,label_non_catalogued,input_files_delimiter,log_file,list_of_user_discarded_features,enco):
    '''
    It allows to obtain all the observations for a specific event which target has a known value,
    it is to say, it is different than int(1). The number of observations to process van be limited

    Parameters:
    :param list<str> vector_with_fullpath_to_input_files: List with the fullpaths to the files where observations are stored
    :param str feature_events_name: Feature that contains the names of the evetns
    :param str event: Event to filter to obtain known observations
    :param str target: Objective feature
    :param int64 maximum_number_of_registers: Maximum number of obervations to concatenate
    :param int label_non_catalogued: Number corresponding to the value of the target for the non-catalogued observations
    :param str input_files_delimiter: Field delimiter for the input files
    :param str log_file: Log to register the information about the current process
    :param list list_of_user_discarded_features: List of variables established by the user to be discarded
    :param str enco: encoding used
    
    :exception Exception: If the feature that contains the names for the events is not available in the dataset
    :exception Exception: if the target feature is not available in the dataset
            
    :return: df_result_for_catalogued_data: Dataframe with recodified data for catalogued observations | empty dataframe if there is not catalogued data             
    :rtype: pandas Dataframe | empty Dataframe
    :return: df_result_for_not_catalogued_data: Dataframe with recodified data for non-catalogued observations | empty dataframe if there is not non-catalogued data             
    :rtype: pandas Dataframe | empty Dataframe
    '''   
        
    name = feature_events_name    
    target = target
    mandatory_features = [name]    
    list_of_catalogued_registers=[] 
    list_of_non_catalogued_registers=[] 
    current_amount_catalogued_observations = 0
    current_amount_non_catalogued_observations = 0
    condition_catalogued_registers = True
    condition_non_catalogued_registers = True
    if(maximum_number_of_registers > 0):
        repg.register_log(log_file,'>> It will be read a maximum number of: ' + str(maximum_number_of_registers) + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    else:
        repg.register_log(log_file,'>> It will be read the maximum number of registers '+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    
    for i in range(len(vector_with_fullpath_to_input_files)):
        if(maximum_number_of_registers != 0):
            condition_catalogued_registers = current_amount_catalogued_observations < maximum_number_of_registers
            condition_non_catalogued_registers = current_amount_non_catalogued_observations < maximum_number_of_registers
        
        '''Checking if maximum number of catalogued and/or not catalogued observations reached'''
        if((i<len(vector_with_fullpath_to_input_files)) and ((condition_catalogued_registers) or (condition_non_catalogued_registers)) ):
            repg.register_log(log_file,'>> Reading csv ' + str(i) + ': ' + vector_with_fullpath_to_input_files[i] + ' ' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
            
            original_data=pd.read_csv(vector_with_fullpath_to_input_files[i],sep=input_files_delimiter)            
            try:                
                original_data = original_data[original_data[name] == event]
            except KeyError:
                raise Exception('Feature selected that should contain the events is not available in datasets')
        
            '''Check if target is available in the dataset'''
            try:
                original_data[target]
            except KeyError:
                raise Exception('Feature selected as target is not available in datasets')
                                    
            '''Separating catalogued and non catalogued observations'''                         
            data_not_catalogued = original_data[original_data[target] == label_non_catalogued]
            data_catalogued = original_data[original_data[target] != label_non_catalogued]
        
            data_not_catalogued = data_not_catalogued.reset_index()
            data_catalogued = data_catalogued.reset_index()
                                
            if(maximum_number_of_registers != 0): # if maximum limit
                current_amount_catalogued_observations = 0
                for ind in range(len(list_of_catalogued_registers)):
                    current_amount_catalogued_observations += len(list_of_catalogued_registers[ind])
                number_of_registers_to_add = (maximum_number_of_registers - (current_amount_catalogued_observations))
                if (number_of_registers_to_add > 0):# if maximum limit was not reached
                    if(len(data_catalogued) < number_of_registers_to_add):            
                        list_of_catalogued_registers.append(data_catalogued)
                    else: 
                        good_data_mod = data_catalogued.iloc[0:number_of_registers_to_add]
                        list_of_catalogued_registers.append(good_data_mod)
                
                current_amount_non_catalogued_observations = 0
                for ind in range(len(list_of_non_catalogued_registers)):
                    current_amount_non_catalogued_observations += len(list_of_non_catalogued_registers[ind])
                number_of_registers_to_add = (maximum_number_of_registers - (current_amount_non_catalogued_observations))
                if (number_of_registers_to_add > 0):
                    if(len(data_not_catalogued) < number_of_registers_to_add):            
                        list_of_non_catalogued_registers.append(data_not_catalogued)
                    else: 
                        bad_data_mod = data_not_catalogued.iloc[0:number_of_registers_to_add]
                        list_of_non_catalogued_registers.append(bad_data_mod)
                        
            else: 
                
                if(not data_catalogued.empty):
                    list_of_catalogued_registers.append(data_catalogued)
                if(not data_not_catalogued.empty):
                    list_of_non_catalogued_registers.append(data_not_catalogued)
    
    '''Checking if valid catalogued data'''
    df_result_for_catalogued_data = pd.DataFrame()
    df_result_for_not_catalogued_data = pd.DataFrame()
    if list_of_catalogued_registers != []:
        data_catalogued=pd.concat(list_of_catalogued_registers)        
        del list_of_catalogued_registers
        coded_list_of_features = []        
        for elemento in data_catalogued.columns:            
            elemento = str(elemento)
            coded_list_of_features.append(elemento)        
        data_catalogued.columns = coded_list_of_features
        data_catalogued[name] = data_catalogued[name].apply(lambda x: str(x))
        head = list(data_catalogued)
        full_list_of_features = remove_feature_from_list('Unnamed: 0',head)
        full_list_of_features = remove_feature_from_list('index',full_list_of_features)
        for variable_to_discard in list_of_user_discarded_features:            
            remove_feature_from_list(variable_to_discard,full_list_of_features)
        df_result_for_catalogued_data = data_catalogued[full_list_of_features]
    else:        
        df_result_for_catalogued_data = pd.DataFrame(columns=mandatory_features) 
        
    if list_of_non_catalogued_registers != []:
        data_not_catalogued=pd.concat(list_of_non_catalogued_registers)        
        del original_data
        del list_of_non_catalogued_registers
        head = list(data_not_catalogued)
        full_list_of_features = remove_feature_from_list('Unnamed: 0',head)
        full_list_of_features = remove_feature_from_list('index',full_list_of_features)
        for variable_to_discard in list_of_user_discarded_features:
            remove_feature_from_list(variable_to_discard,full_list_of_features)        
        df_result_for_not_catalogued_data = data_not_catalogued[full_list_of_features]
    else:        
        df_result_for_not_catalogued_data = pd.DataFrame(columns=mandatory_features)
    
    return df_result_for_catalogued_data,df_result_for_not_catalogued_data


def get_two_subsets_using_random_split(dataframe,percentaje):
    '''
    It allows to obtain two new dataframes train and test dataframe according to the 
    specified percentaje

    Parameters:
    :param pandas_dataframe dataframe: Dataframe to split between train and test data
    :param int percentaje: Percentaje to divide the original dataframe
        
    
    :return: A new tuple of dataframes for train and test, if no valid observations, an empty Dataframe is returned.
    :rtype: tuple(list<str>,pandas Dataframe) | empty Dataframe
    '''
    
    msk = np.random.rand(len(dataframe))
    bound=np.percentile(msk,percentaje*100)
    msk=msk<bound
    train=dataframe[msk]
    test=dataframe[~msk]
    return(train,test)


def split_train_test_datasets(df_catalogued_data,target,percentaje):   
    '''
    It allows to obtain two new dataframes of observations according to a percentaje

    Parameters:
    :param pandas_dataframe df_catalogued_data: dataframe to split between train and test data
    :param str target: objective feature
    :param int percentaje: percentaje to divide the original dataframe
        
    
    :return: train_data, dataframe with observations to train the models
    :rtype: pandas Dataframe   
    :return: test_data, dataframe with observations to test the models
    :rtype: pandas Dataframe   
    '''    
    
    current_different_targets = list(set(df_catalogued_data[target].values))    
    train_data = []
    test_data = []        
    
    for current_target in current_different_targets:        
        df_registers_current_target = df_catalogued_data[df_catalogued_data[target] == current_target]
        current_train_data, current_test_data=get_two_subsets_using_random_split(df_registers_current_target, percentaje)
        if(len(current_train_data) < len(current_test_data)):
            aux = current_test_data.copy()
            current_test_data = current_train_data.copy()            
            current_train_data = aux.copy()            
        train_data.append(current_train_data)
        test_data.append(current_test_data)
        
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)
    
    train_data=train_data.reset_index()
    test_data=test_data.reset_index()
         
    train_data=train_data.drop('index',axis=1)
    test_data=test_data.drop('index',axis=1)

    return train_data,test_data


def get_models_to_train(learning,list_of_models):
    '''
    It allows to obtain the full list of available models to apply

    Parameters:
    :param learning: type of learning, supervised or unsupervised
    :param list_of_models: list of available models for the type of learning
            
    :return: models_to_apply,dictionary with model to apply
    :rtype: dict<learning:model>
    '''
    
    dictionary_with_learnings_related_models = datr.get_dictionary_with_learnings_related_to_models()
    
    models_to_apply = [] #list with the names of the models that will be applied to the event
    for current_model in list_of_models:
        if current_model == 0: #it means that all the supervised/unsupervised models available will be applied
            for key in dictionary_with_learnings_related_models[learning]:
                if(key != 0):
                        selected_model = dictionary_with_learnings_related_models[learning][key]
                        models_to_apply.append(selected_model)
        else:
            selected_model = dictionary_with_learnings_related_models[learning][current_model]
            models_to_apply.append(selected_model)
        
    return models_to_apply


def angle_two_points(a,b):
    '''
    It allows to calculate the angle between a line and the horizontal axis
    Parameters:
    :param list a: a list of two coordinates that defines one point
    :param list b: a list of two coordinates that defines one point
    
    :return alpha: angle in radians
    :rtype float
    '''
    
    xa=float(a[0])
    ya=float(a[1])
    xb=float(b[0])
    yb=float(b[1])
    tangent=(yb-ya)/float(xb-xa)
    alpha=np.arctan(tangent)
    return(alpha)

def angle_two_lines(ts,ss,cvs):
    '''
    It allows to calculate the mean convergence velocity
    Parameters:
    :param list ts: horizontal axis points
    :param list ss: points corresponding to one path
    :param list cvs: points corresponding to one path
    
    :return alpha: angle in radians
    :rtype float
    '''
    ass=[ts[0],ss[0]]
    bss=[ts[-1],ss[-1]]
    acv=[ts[0],cvs[0]]
    bcv=[ts[-1],cvs[-1]]
    alphass=angle_two_points(ass,bss)
    alphacv=angle_two_points(acv,bcv)
    alpha=alphacv-alphass
    return(alpha)
    
 
def transformation_learning_curve(ts,ss,cvs):
    '''
    It allows to calculate plain transformation
    Parameters:
    :param list ts: horizontal axis points
    :param list ss: points corresponding to one path
    :param list cvs: points corresponding to one path
    
    :return ts: horizontal axis points, ss: points corresponding to one transformed path, cvs: points corresponding to one transformed path
    :rtype list<list<float,float>>    
    '''
    
    new_ss=[]
    new_cvs=[]
    for i in range(len(ts)):
        new_ss.append(ss[i]+(1-ss[i])/float(10))
        new_cvs.append(cvs[i]-cvs[i]/float(10))
    return(ts,new_ss,new_cvs)


def weight_for_decision_function(ts,ss,cvs,cm,w,main_metric,compute_lc=True,penalize_falses=False):
    '''It allows to calculate plain transformation
    Parameters:
    :param list ts: horizontal axis points
    :param list ss: points corresponding to one path
    :param list cvs: points corresponding to one path
    :param numpy-array cm: confusion matrix
    :param list: weighted matrix
    :param str main_metric: main metric (acc or mcc) to calculate the decision index
    
    :return computed_weight: score for the current model
    :rtype float
    '''
	
    computed_weight=0
    cw=0 
    cwfin=0
    total=cm/float(np.sum(cm))
    
    #Do not penalize FP and FN
    if (not penalize_falses):
        for i in range(len(cm)):
            for j in range(len(cm)):
                cw=cw+total[i][j]*w[i][j]
        cwfin=cw
        
    else:# penalize FP and FN
        for i in range(len(cm)):
            for j in range(len(cm)):
                if(i==j):
                    cw=cw+total[i][j]*w[i][j]
                else:
                    cw=cw-total[i][j]*w[i][j]
        cwfin=cw
    
    if(compute_lc):
        nts=transformation_learning_curve(ts,ss,cvs)[0]
        nss=transformation_learning_curve(ts,ss,cvs)[1]
        ncvs=transformation_learning_curve(ts,ss,cvs)[2]
        alpha=float(angle_two_lines(nts,nss,ncvs)) #radians
        pi_final=float(np.abs(nss[-1]-ncvs[-1]))
        pi_zero=float(np.abs(nss[0]-ncvs[0]))
        computed_weight=main_metric*10 + float(cwfin) - np.abs(float(pi_final)) - 2*(np.pi/2-np.abs(alpha))*np.abs(float(pi_zero))/np.pi
    else:
        computed_weight=main_metric*10 + float(cwfin)
    return(computed_weight)
    
    
def random_selection(df,percentaje):
    '''
    It allows to obtain a sud-dataframe of the original dataframe based on a percentaje

    Parameters:
    :param pandas_dataframe df: dataframe to get the sub-dataframe    
    :param int percentaje: percentaje to divide the original dataframe
        
    
    :return: df_selection, dataframe with the sub-dataframe with the specified percentaje of samples
    :rtype: pandas Dataframe   
    '''    

    msk = np.random.rand(len(df))
    bound=np.percentile(msk,percentaje)
    msk=msk<bound
    df_selection=df[msk]
    return(df_selection)


def random_selection_same_distribution(df,target,percentaje):
    '''
    It allows to obtain a sud-dataframe of the original dataframe based on a percentaje and on a 
    specific feature to get the distribution.

    Parameters:
    :param pandas_dataframe df: dataframe to get the sub-dataframe    
    :param str target : name of the feature in which is based the percentaje of the distribution
    :param int percentaje: percentaje to divide the original dataframe
        
    
    :return: df_random_selection, dataframe with the sub-dataframe with the specified percentaje of samples
    :rtype: pandas Dataframe   
    '''    
    
    target_values=list(set(list(df[target].values)))
    df_per_value=[]
    for value in target_values:
        df_value=df[df[target]==value]
        df_value=random_selection(df_value,percentaje)
        df_per_value.append(df_value)
        del df_value
    df_random_selection=pd.concat(df_per_value)
    return(df_random_selection.reset_index())


'''Functions related to the selection of the best model and the predictions part'''

def get_dictionary_of_models_score_time(report_dict_event,generic_list_of_models):
    '''
    It allows to get the dictionary of models with their score and time
    
    Parameters:
    :param dict report_dict_evento: current information about the event in dict format
    :param list generic_list_of_models: list with the avilable models to be apply    
    
    :return dictionary_models_score_time, dictionary wit the models and their current score and time
    :rtype list<list<str,score,time>>  
    '''
    
    dictionary_models_score_time = {}
    for model in generic_list_of_models:
        if (model in report_dict_event):            
            index_dec_modelo = round(float(report_dict_event[model][glod.get_decision_index_key()]),4)
            training_elapsed_time = report_dict_event[model][glod.get_time_parameters_key()][glod.get_time_train_finish_key()] - report_dict_event[model][glod.get_time_parameters_key()][glod.get_time_train_init_key()]
            dictionary_models_score_time[model] = {glod.get_current_score_key():index_dec_modelo,glod.get_current_time_key(): training_elapsed_time}
            
    return dictionary_models_score_time


def order_models_by_score_and_time(report_dict_event,generic_list_of_models):
    '''
    It allows to get the list of models ordered by score and time
    
    Parameters:
    :param dict report_dict_event: Current information about the event in dict format
    :param list generic_list_of_models: list with the avilable models to be applied 
    
    :return resultant_list_ordered_by_models_time: List of models ordered by score and time
    :rtype list<list<str,score,time>>  
    '''    
    
    dictionary_models_scores_time = get_dictionary_of_models_score_time(report_dict_event,generic_list_of_models)
    resultant_list_ordered_by_models_time = []    
    
    while (dictionary_models_scores_time != {}):                
        maximum_score_available = ''
        list_models_maximum_score = []
        for model in dictionary_models_scores_time:
            if(maximum_score_available == ''):
                maximum_score_available = dictionary_models_scores_time[model][glod.get_current_score_key()]
            else:
                current_score = dictionary_models_scores_time[model][glod.get_current_score_key()]
                if(current_score > maximum_score_available):
                    maximum_score_available = current_score
                    
        '''Get models with maximum available score'''
        for model in dictionary_models_scores_time:
            score_to_check = dictionary_models_scores_time[model][glod.get_current_score_key()]
            if(score_to_check == maximum_score_available):
                list_models_maximum_score.append(model)
        
        '''Get list with models that share maximun score'''
        if(len(list_models_maximum_score) > 1):
            '''Checking times to get the best'''
            cp_models_max_score = copy.deepcopy(list_models_maximum_score)
            while(list_models_maximum_score != []):
                best_time = ''
                fastest_model_name = ''
                for model in list_models_maximum_score:
                    if (best_time == ''):
                        best_time = dictionary_models_scores_time[model][glod.get_current_time_key()]
                        fastest_model_name = model
                        
                    else:
                        current_time = dictionary_models_scores_time[model][glod.get_current_time_key()]
                        if(current_time < best_time):
                            best_time = current_time
                            fastest_model_name = model
                
                resultant_list_ordered_by_models_time.append([fastest_model_name,maximum_score_available,best_time])
                list_models_maximum_score.remove(fastest_model_name)
            for modelo_eliminar in cp_models_max_score:
                del dictionary_models_scores_time[modelo_eliminar]                
            
        else:
            model = list_models_maximum_score[0]            
            time_for_current_model = dictionary_models_scores_time[model][glod.get_current_time_key()]
            resultant_list_ordered_by_models_time.append([model,maximum_score_available,time_for_current_model])
            del dictionary_models_scores_time[model]
            
    return resultant_list_ordered_by_models_time


def get_model_with_highest_decision_index(report_dict,generic_list_of_models):
    '''
    It allows to get the model with the highest decision index
    
    Parameters:
    :param dict report_dict: Current information about the event in dict format
    :param list generic_list_of_models: List with the avilable models to be applied   
    
    :return model_max_dec, name of the best model
    :rtype str
    :return max_dec_index, maximun decision index
    :rtype float
    '''
    
    ordered_list_model_score_time = order_models_by_score_and_time(report_dict,generic_list_of_models)
    
    model_max_dec_index = ordered_list_model_score_time[0][0]
    max_dec_index = ordered_list_model_score_time[0][1]
    print ("best classifier: " +  model_max_dec_index)
    return model_max_dec_index,max_dec_index


def add_entry_for_event_in_prediction_dictionary(prediction_dictionary,event):
    '''
    It allows to add a new entry for the event in the prediction dictionary
    
    Parameters:
    :param dict prediction_dictionary: current information about events and 
    :param event: name of the event to add
    
    :return prediction_dictionary, dictionary with events and prediction models info
    :rtype dict
    '''
    
    if(event not in prediction_dictionary):
        prediction_dictionary[event] = {}    
    return prediction_dictionary


'''def get_original_list_features(list_features_derived_and_original,reserved_character):
    
    It allows to retreieve the list wiht the original names of derived features
    
    Parameters:
    :param list list_features_derived_and_original: list of the features
    :param reserved_character: special character to obtain derived fatures
    
    :return original_list_of_features, list of the original features
    :rtype dict
    
    
    original_list_of_features = []
    for feature in list_features_derived_and_original:
        if reserved_character in feature:
            feature = feature.split(reserved_character)
            feature = feature[0]
            original_list_of_features.append(feature)
        else:
            original_list_of_features.append(feature)
        original_list_of_features = sorted(list(set(original_list_of_features)))
    return original_list_of_features'''


def update_prediction_models_original_features_for_event_in_prediction_dictionary(prediction_dictionary,event,target,model_characteristics,list_of_parameters):
    '''
    It allows to store in the predictino dictionary the list with original features
    Parameters:
    :param dict prediction_dictionary: Current information about events and related prediction models
    :param event: Name of the event to add to the dictionary
    :param target: Name of the target to add to the dictionary for the event
    :param model_characteristics: Characteristics of the model (type of learning, path to the pkl...) to be stored in the dictionary
    :param reserved_character: Character used to construct the derived features
    :param list_of_parameters: List with the names of the characteristics to be registere.
        
    
    :return prediction_dictionary: Dictionary with events and prediction models info
    :rtype dict
    '''
    
    prediction_dictionary[event][target] = {}
    prediction_dictionary[event][target][list_of_parameters[0]] = model_characteristics[0]
    prediction_dictionary[event][target][list_of_parameters[1]] = model_characteristics[1]
    prediction_dictionary[event][target][list_of_parameters[2]] = model_characteristics[2]
    prediction_dictionary[event][target][list_of_parameters[3]] = model_characteristics[3]        
    
    return prediction_dictionary


import re
def natsorted(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

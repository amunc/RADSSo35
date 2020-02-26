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

import time
import datetime
import os
import pandas as pd
import traceback
import configparser as conp
from sklearn.externals import joblib

import auxiliary_functions as auxf
import global_definitions as glod
import reports_generation as repg
import metrics as metr

import warnings
warnings.filterwarnings("ignore")

def prediction_function(BASE_PATH):
       
    ''' Step 0: Reading configuration parameters and creating log files'''
    '''Creating variable that parses the configuration file. If the file is
    not found, an exception is thrown and finishes the execution'''
        
    path_to_configuration_file =os.path.join(BASE_PATH,glod.get_config_parser_name())
    config_parser = conp.ConfigParser()
    config_parser.optionxform = str
    enco = glod.get_encoding()

    if(os.path.exists(path_to_configuration_file)):
        config_parser_file = open(path_to_configuration_file, encoding=enco)
        config_parser.readfp(config_parser_file)
    else:
        raise Exception('Configuration file (conf.ini) was not found')
    
    
    logs_section = glod.get_log_section_name()
    auxiliary_data_section = glod.get_auxiliary_section_name()
    input_data_section = glod.get_input_section_name()
    prediction_section = glod.get_prediction_section_name()
    

                
    '''Creating log files'''
    log_path = os.path.join(BASE_PATH, config_parser.get(logs_section, glod.get_log_directory_name()))
    execution_log_path = os.path.join(log_path, config_parser.get(logs_section, glod.get_prediction_log_execution_name()) +'.' + glod.get_log_files_extension())
    time_log_path = os.path.join(log_path, config_parser.get(logs_section,glod.get_prediction_log_time_execution_name()) +'.' + glod.get_log_files_extension())
    ruta_modelos_prediccion = config_parser.get(prediction_section, glod.get_path_to_prediction_models_name())
    auxf.create_directory(log_path)
    
    step_init_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([execution_log_path],'>>>>>>Prediction Phase <<<<<<<<   \n'+ step_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",0,enco)
    repg.register_log([execution_log_path],'>>>> Step 0 - Reading parameters from conf.ini \n'+ step_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    repg.register_log([time_log_path],'>>>>Step 0 starts:\n',0,enco)
    
    '''Reading from conf.ini necessary variables for the prediction phase'''
    extension = glod.get_input_files_extension()
    name= config_parser.get(input_data_section,glod.get_event_name_feature_name())
    
    observation_number = config_parser.get(input_data_section,glod.get_obsnumber_parameter_name())
    input_files_delimiter_not_catalogued_data = config_parser.get(prediction_section,glod.get_delimiter_non_catalogued_data_name())
    input_files_delimiter_not_catalogued_data = input_files_delimiter_not_catalogued_data.replace('\'','')
    label_non_catalogued_data = int(config_parser.get(input_data_section,glod.get_non_catalogued_label_name()))
    
    maximum_number_of_files_to_catalogue= int(config_parser.get(prediction_section,glod.get_number_of_files_parameter_name()))
    path_to_directory_with_input_files_to_catalogue = os.path.join(BASE_PATH, config_parser.get(prediction_section,glod.get_name_directory_to_input_files_catalogue()))
    
    step_finish_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([execution_log_path],'>>>> Step 0 ends \n'+ step_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    repg.register_log([time_log_path],'>>>>Step 0 - Reading parameters from conf.ini total elapsed time :' +str(step_finish_time -step_init_time) +'\n','',enco)
    
    
    ''' Step 1: Reading observations from files and concatenating them '''
    step_init_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([execution_log_path],'>>>>Step 1 Loading observations from files into dataframes \n'+ step_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    repg.register_log([time_log_path],'>>>>Step 1 starts:\n','',enco)    
    vector_fullpaths_to_input_files_with_observations_to_catalogue= auxf.get_all_files_in_dir_with_extension(path_to_directory_with_input_files_to_catalogue, maximum_number_of_files_to_catalogue,extension)    
                
    ''' Path to file with relevant variables '''
    auxiliary_directory_filename = config_parser.get(auxiliary_data_section,glod.get_auxiliary_directory_parameter_name())
    path_to_directory_auxiliary_files = os.path.join(BASE_PATH, config_parser.get(auxiliary_data_section,glod.get_auxiliary_directory_parameter_name()))    
    
    
    report_dict = {glod.get_logo_key(): "'" + os.path.join(path_to_directory_auxiliary_files,glod.get_logo_name()) + "'"}
                
    ''' Substep 1.1 - Reading input files '''    
    substep_init_time = datetime.datetime.fromtimestamp(time.time())
    list_registers_to_catalogue = []
    repg.register_log([execution_log_path],'>>>>Step 1.1 \n','',enco)
    for i in range(len(vector_fullpaths_to_input_files_with_observations_to_catalogue)):        
        repg.register_log([execution_log_path],'>>Reading Csv to predict number '+ str(i)+': '+ vector_fullpaths_to_input_files_with_observations_to_catalogue[i] + '\n','',enco)
        print ("To catalogue : ", vector_fullpaths_to_input_files_with_observations_to_catalogue[i])
        print ("\n")
        original_data=pd.read_csv(vector_fullpaths_to_input_files_with_observations_to_catalogue[i],sep=input_files_delimiter_not_catalogued_data)
        list_registers_to_catalogue.append(original_data)
                
    substep_finish_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([time_log_path],'>>>>Subtep 1.1 - reading csv total elapsed time: '+ str(substep_finish_time-substep_init_time) +'\n','',enco)
    repg.register_log([execution_log_path],'>>>>Subtep 1.1 - reading csv total elapsed time: '+ str(substep_finish_time-substep_init_time) +'\n','',enco)    
    
    if(list_registers_to_catalogue == list()):
        repg.register_log([time_log_path],'>>>> Prediction process finished: Observations were not found '+ str(substep_finish_time-substep_init_time) +'\n','',enco)
        repg.register_log([execution_log_path],'>>>> Prediction process finished: Obervations were not found '+ str(substep_finish_time-substep_init_time) +'\n','',enco)    
        print ('>>>> Prediction process finished: Obervations were not found ')
    
    ''' Substep 1.2 - Concatenating read csv'''        
    substep_init_time = datetime.datetime.fromtimestamp(time.time())
    df_data_to_catalogue = pd.concat(list_registers_to_catalogue)
    reco_pandas_features = []
    for feature in df_data_to_catalogue.columns:
        reco_pandas_features.append(feature)       
    df_data_to_catalogue.columns = reco_pandas_features

    try:
        df_data_to_catalogue[name]
    except Exception as e:
        repg.register_log([execution_log_path],'>> An Eception has happened: Incorrect name of feature with events '+ str(e) + ' ' +datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + '\n','',enco)
        print ('>> An Exception has happened, check configuration file: Incorrect name of feature with events "'+ str(e) + '"')
        error_trace =  "Full trace:\n" + str(traceback.format_exc())
        repg.register_log([execution_log_path], error_trace,'',enco)
        raise Exception(e)        
        
    ''' Erasing indexes introduced by pandas, if any '''         
    if 'index' in df_data_to_catalogue.columns:
        df_data_to_catalogue = df_data_to_catalogue.drop('index',axis=1)
    if 'Unnamed: 0' in df_data_to_catalogue.columns:
        df_data_to_catalogue = df_data_to_catalogue.drop('Unnamed: 0',axis=1)
        
    substep_finish_time = datetime.datetime.fromtimestamp(time.time())
    step_finish_time = datetime.datetime.fromtimestamp(time.time())
    total_time_step_1 = step_finish_time - step_init_time
    repg.register_log([time_log_path],'>>>> Substep 1.2 - Loading observations from files into dataframes total elapsed time: ' + str(substep_finish_time-substep_init_time)  + "\n",'',enco)
    repg.register_log([execution_log_path],'>>>>Substep 1.2 ends '+ step_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    repg.register_log([time_log_path],'>>>> Step 1 - Reading and concatenating csv into dataframe total elapsed time: ' + str(total_time_step_1)  + "\n",'',enco)
    repg.register_log([execution_log_path],'>>>>Step 1 ends '+ step_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    
    
    
    ''' Step 2: Reading prediction models dictionary and preloading best pkl models'''
    step_init_time = datetime.datetime.fromtimestamp(time.time())    
    repg.register_log([execution_log_path],'>>>>Step 2 Reading models dict and preload best pkl models \n'+ step_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    repg.register_log([time_log_path],'>>>>Step 2 starts:\n','',enco)
    
    '''Getting dictionary features in order to recodify and the events to catalogue'''    
    substep_init_time = datetime.datetime.fromtimestamp(time.time())
    dic_event_model,handler=auxf.open_dictionary_pickle_format_for_reading(ruta_modelos_prediccion)
    
    substep_finish_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([time_log_path],'>>>> Substep 2.1 - Reading dictionary with models total elapsed time: ' + str(substep_finish_time-substep_init_time)  + "\n",'',enco)
    repg.register_log([execution_log_path],'>>>>Substep 2.1 Reading dictionary with models ends '+ substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    
    
    '''Preloading models in memory'''
    substep_init_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([execution_log_path],'>>>>Substep 2.2 - Preloading best pkl models \n'+ substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    list_features_to_catalogue = []
    dictionary_of_events_preloaded_models = {}
    print('''
                                          Events, target and predictions models
             ##############################################################################################
             ##                                                                                          ##''')
    for event in dic_event_model.keys():
        dictionary_of_events_preloaded_models[event] ={}
        try:        
            for target_trained in dic_event_model[event]:
                dictionary_of_events_preloaded_models[event][target_trained] = {}            
                best_model = joblib.load(dic_event_model[event][target_trained][glod.get_model_path_key()])
                print ("\t\t\t\t",dic_event_model[event][target_trained][glod.get_model_path_key()])
                dictionary_of_events_preloaded_models[event][target_trained][glod.get_best_model_key()] = best_model
                list_features_to_catalogue+=dic_event_model[event][target_trained][glod.get_current_features_key()]
        except Exception as e:
            print ('''
             ##                                                                                          ##
             ##############################################################################################''')
            print('The pkl neccesary for the prediction of the observations of ' + event + ' was not found ')
            repg.register_log([execution_log_path],'The pkl neccesary for the prediction of the observations of ' + event + ' was not found ' + substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
            raise Exception(e)
    print('''
             ##                                                                                          ##
             ##############################################################################################''')
                     
    substep_finish_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([time_log_path],'>>>>Substep 2.2 - Preloading best pkl models total elapsed time: '+ str(substep_finish_time-substep_init_time) + '\n','',enco)
    repg.register_log([execution_log_path],'>>>>Substep 2.2 - Preloading best pkl models ends \n'+ substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)                     
    step_finish_time = datetime.datetime.fromtimestamp(time.time())
    total_time_step_2 = step_finish_time - step_init_time
    repg.register_log([time_log_path],'>>>> Step 2 - Reading models dict and preload best pkl models total elapsed time: ' + str(total_time_step_2)  + "\n",'',enco)    
    repg.register_log([execution_log_path],'>>>>Step 2 ends \n'+ step_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    
    ''' Step 3: Classifying observations usin preloaded models '''
    
    maximum_number_of_observations_to_catalogue= len(df_data_to_catalogue)
    
    step_init_time = datetime.datetime.fromtimestamp(time.time())
    substep_init_time = datetime.datetime.fromtimestamp(time.time())    
    
    repg.register_log([time_log_path],'>>>> Step 3 starts \n','',enco)
    repg.register_log([execution_log_path],'>>>>Step 3 - Predicting targets using best models \n'+ substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    repg.register_log([execution_log_path],'>>>>Substep 3.1 - Preparing global dataframe of results \n'+ substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    observations=df_data_to_catalogue.iloc[0:maximum_number_of_observations_to_catalogue]    
    events_to_predict = list(set(observations[name].values))
        
    #target to predict
    target_to_predict = config_parser.get(prediction_section, glod.get_target_parameter_name())
    
    #column for predictions
    prediction_column = target_to_predict + '_pred'
    
    
    df_global_predictions = pd.DataFrame(data=[],columns=[observation_number,prediction_column])    
    
    substep_finish_time = datetime.datetime.fromtimestamp(time.time())
    repg.register_log([time_log_path],'>>>>Subtep 3.1 - Preparing global dataframe of results total elapsed time: '+ str(substep_finish_time-substep_init_time) + "\n",'',enco)
    repg.register_log([execution_log_path],'>>>>Substep 3.1 ends \n'+ substep_finish_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
    total_number_predicted_observations = 0
    final_list_events_to_predict = []
    for event in events_to_predict:  
        substep_init_time = datetime.datetime.fromtimestamp(time.time())
        repg.register_log([time_log_path],'>>>>Subtep 3.2 - Predicting targets for event '+event + ' \n','',enco)
        repg.register_log([execution_log_path],'>>>>Substep 3.2 - Predicting targets for event '+ event + substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
        df_event = observations[observations[name]==event]          
        df_event_obsnumber = pd.DataFrame(data=df_event[observation_number].values,columns=[observation_number])        
                
        try:
            dic_event = dictionary_of_events_preloaded_models[event]            
            total_number_predicted_observations+=len(df_event)
            
            if target_to_predict not in df_event.columns:
                repg.register_log([execution_log_path],'>> ###Error: The target ' + target_to_predict + ' does not exist in the dataset of the event ' + event + '\n\n','',enco)
                raise Exception('>> ###Error: The target ' + target_to_predict + ' does not exist in the dataset of the event ' + event)                 
                        
            if target_to_predict in dic_event:
                repg.register_log([execution_log_path],'>> The event ' + event + ' (with ' + str(len(df_event)) + ' observations), has a model for predict target labels \n','',enco)                
                features_event=dic_event_model[event][target_to_predict][glod.get_current_features_key()] 
            
                model_event = dictionary_of_events_preloaded_models[event][target_to_predict][glod.get_best_model_key()] #se referencia al modelo 
                predictions=model_event.predict(df_event[features_event])
                df_event_obsnumber[prediction_column] = predictions                                
                
                recatalogued_predictions = []                                
                if dic_event_model[event][target_trained][glod.get_learning_key()] == glod.get_unsupervised_name():                    
                    for pred in predictions:                        
                        recatalogued_predictions.append(dic_event_model[event][target_trained][glod.get_reasignment_dict_key()][pred])
                    predictions = recatalogued_predictions
                df_event_obsnumber[prediction_column] = predictions
                df_event_obsnumber[name] = event
            
            else:
                repg.register_log([execution_log_path],'>> The event ' + event + ' (with ' + str(len(df_event)) + ' observations), has not models for predicting target (' + target_to_predict +'). Taking original prediction \n','',enco)
                total_number_predicted_observations+=len(df_event)        
                df_event_obsnumber[prediction_column] = df_event[target_to_predict].values
            
            final_list_events_to_predict.append(event)
            
        except Exception as excep: #no predictions models
            repg.register_log([execution_log_path],'>> The prediction process has been aborted ' + str(excep) +'\n','',enco)
            #raise Exception(e)
            
                        
        df_global_predictions = pd.concat([df_global_predictions,df_event_obsnumber])
        df_global_predictions[observation_number] = df_global_predictions[observation_number].apply(int)
        
        substep_finish_time = datetime.datetime.fromtimestamp(time.time())
        repg.register_log([time_log_path],'>>>>Substep 3.2 - Predicting targets for event '+ event + ' total elapsed time: ' + str(substep_finish_time-substep_init_time) + "\n",'',enco)
        repg.register_log([time_log_path],'>>>>Substep 3.2 - Estimated elapsed time predicting one observation for event ' + event +': ' + str(float((substep_finish_time-substep_init_time).total_seconds())/float(len(df_event))) + "\n",'',enco)
        repg.register_log([execution_log_path],'>>>>Substep 3.2 - Predicting targets for event '+ event + ' ends ' + substep_init_time.strftime('%Y-%m-%d %H:%M:%S') + "\n",'',enco)
        
    type_observation_number = df_global_predictions[observation_number].dtypes
    observations[observation_number] = observations[observation_number].astype(type_observation_number)
    observations = pd.merge(observations, df_global_predictions, on=[observation_number,name])
    
    for not_proc_event in set.difference(set(events_to_predict),set(final_list_events_to_predict)):
        repg.register_log([execution_log_path],'>> WARNING: Event ' + not_proc_event + ' has not models, but validation/unkown samples dataset was provided\n','',enco)
        print ("**WARNING**: Event " + not_proc_event + " has not models, but validation/unkown samples dataset was provided")
        
    for event in events_to_predict:        
        print ('\n-> Event: ', event)
        df_event = observations[observations[name] == event]
        path_to_predicted_data_root = 'Prediction_models'
        path_to_predicted_data = config_parser.get(prediction_section,glod.get_path_predicted_data_key())
        
        #Accuracy print      
        if (event in dic_event_model and  target_to_predict in dic_event_model[event]):
            report_dict[event] = {target_to_predict:{}}            
            print ('\t\tObservations with known target ', len(df_event[df_event[target_to_predict] !=  label_non_catalogued_data]))
            print ('\t\tObservations with unknown target ', len(df_event[df_event[target_to_predict] == label_non_catalogued_data]))
            df_observaciones = df_event[df_event[target_to_predict] != label_non_catalogued_data]
            total_obs = len(df_observaciones)
            
            #computing confusion matrix
            pred_labels = list(df_observaciones[prediction_column].values)
            true_labels = list(df_observaciones[target_to_predict].values)            
            
            if( pred_labels != [] and true_labels != []):
                df_observaciones_temp = df_observaciones[df_observaciones[target_to_predict] == df_observaciones[prediction_column]]
                total_aciertos = len(df_observaciones_temp)

                confusion_matrix=metr.get_confusion_matrix(true_labels,pred_labels,sorted(list(set(df_observaciones[target_to_predict].values))))
                confusion_matrix_name='confusion_matrix_'+ event +'_' + target_to_predict
                metr.save_confusion_matrix(confusion_matrix,sorted(list(set(df_observaciones[target_to_predict].values))),os.path.join(path_to_predicted_data_root,confusion_matrix_name),'png')
            
                report_dict[event][target_to_predict][glod.get_best_model_key()] = str(dictionary_of_events_preloaded_models[event][target_to_predict][glod.get_best_model_key()])
                report_dict[event][target_to_predict]['Correct'] = total_aciertos
                report_dict[event][target_to_predict]['Total'] = len(df_event[df_event[target_to_predict] !=  label_non_catalogued_data])
                report_dict[event][target_to_predict][glod.get_accuracy_parameter_name()] = float(float(total_aciertos)/float(len(df_event[df_event[target_to_predict] != label_non_catalogued_data])))
                report_dict[event][target_to_predict]['target_to_predict_cm'] = os.path.join(path_to_predicted_data_root,confusion_matrix_name) + '.png'
                report_dict[event][target_to_predict]['Predicted'] = len(df_event[df_event[target_to_predict] ==  label_non_catalogued_data])
            
                        
        else:
            total_obs = 0   

        if(total_obs != 0):
            repg.register_log([time_log_path],'>>>>Substep 3.2 Extra - Accuracy of the model for event '+ event + ' and target '+ target_to_predict +'('+str(float(total_aciertos))+'/'+str(float(total_obs)) +'): ' + str(float(total_aciertos)/float(total_obs)) + "\n",'',enco)
            repg.register_log([execution_log_path],'>>>>Substep 3.2 Extra - Accuracy of the model for event '+ event + ' and target '+ target_to_predict +'('+str(float(total_aciertos))+'/'+str(float(total_obs)) +'): ' + str(float(total_aciertos)/float(total_obs)) + "\n",'',enco)
        else:
            repg.register_log([time_log_path],'>>>>Substep 3.2 Extra - Accuracy of the model for event '+ event + ' and target '+ target_to_predict +': not calculated (no observations found) \n','',enco)
            repg.register_log([execution_log_path],'>>>>Substep 3.2 Extra - Accuracy of the model for event '+ event + ' and target '+ target_to_predict +': not calculated (no observations found) \n','',enco)            
            
            
        print('''
                                            Clasification for known targets results
             ##############################################################################################'
             ##                                                                                          ##''')
        if(total_obs != 0):
            print ('\t\t\t\tCorrect predictions performed for ' + target_to_predict +' of event ' + event + ': ' + str(total_aciertos) + '/' + str(total_obs))
        else:
            print ('\t\t\t\tNo predictions performed for severity of event ' + event)
            
        print ('\t\t\t\tCheck output data at: ', path_to_predicted_data)
        print ('\t\t\t\tCheck predictions log for accuracy summary and more information')
        print ('\n\t\t\t\t\t\tThanks for using RADSSo')
        print ('''             
             ##                                                                                          ##
             ##############################################################################################''')


        #Determinamos cuantas predicciones sobre datos desconocidos se han realizado
        if (event in dic_event_model and  target_to_predict in dic_event_model[event]):
            df_observaciones = df_event[df_event[target_to_predict] == label_non_catalogued_data]                                
            total_obs = len(df_observaciones)            
    
        print ('''
                                                Prediction of unknown target results
             ##############################################################################################'
             ##                                                                                          ##''' )
        print ('\t\t\t\tTotal of predictions performed for ' + target_to_predict+ ': ', str(total_obs))
        print ('\t\t\t\tCheck output data at: ', path_to_predicted_data)
        print ('\t\t\t\tCheck predictions log for accuracy summary and more information')
        print ('\n\t\t\t\t\t\tThanks for using RADSSo')
        print ('''             
             ##                                                                                          ##
             ##############################################################################################\n''')
        try:
            if (event in report_dict):
                if( pred_labels != [] and true_labels != []):
                    repg.create_report_prediction(report_dict, [event, target_to_predict], auxiliary_directory_filename, 'Prediction_models', enco)
        except Exception as e:
            print (''''**********************
                   ****Critical Exception****
                   **************************''')
            print(e)
        
    step_finish_time = datetime.datetime.fromtimestamp(time.time())
    total_time_step_3 = step_finish_time- step_init_time
    print ('\n\n--- Total Elapsed time --- ' + str(total_time_step_3))
        
    repg.register_log([time_log_path],'>>>>Step 3 - Predicting using best models total elapsed time: ' + str(total_time_step_3)  + "\n",'',enco)

    repg.register_log([time_log_path],'>>>> Number of observations processed by second ' + str(float(total_number_predicted_observations)/float(total_time_step_3.total_seconds())) + "\n",'',enco)
    repg.register_log([time_log_path],'>>>> Number of seconds by prediction ' + str(float(total_time_step_3.total_seconds())/float(total_number_predicted_observations)) + "\n",'',enco)

    repg.register_log([time_log_path],'>>>> Recodification and Prediction Phase - total elapsed time: ' + str(total_time_step_1 + total_time_step_2 + total_time_step_3)  + "\n",'',enco)
    
    observations.to_csv(path_to_predicted_data,sep=input_files_delimiter_not_catalogued_data,encoding=enco)
    return()

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
try:
    prediction_function(BASE_PATH)
except Exception as excep:
    print(excep)

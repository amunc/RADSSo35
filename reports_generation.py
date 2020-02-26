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


import base64
import logging
import operator
import os
import codecs
import numpy as np

import xhtml2pdf.pisa as pisa
from jinja2 import Environment, FileSystemLoader

import auxiliary_functions as auxf
import global_definitions as glod

class PisaNullHandler(logging.Handler):
    '''Managing pisa logging'''
    def emit(self, record):
        pass


def encode_image(path_to_image):
    '''This function allows to encode an iage in base64
    Parameters:
        path_to_image: string with the path to the image to encode

    return: string with the image encoded in base64
    '''

    with open(path_to_image, glod.get_readbyte_mode()) as image_file:
        image_read = image_file.read()
        image_64_encode = base64.b64encode(image_read)
        image_encoded = image_64_encode.decode(glod.get_encoding())
        return 'data:image/png;base64,' + image_encoded


def register_log(array_rutas_ficheros, mensaje, opcion, enco):
    '''
    This function allows to register information in the logs

    Parameters:
    :param list array_rutas_ficheros: list with the path to the lgos files
    :param str mensaje: Message to register in the log files
    :param str/int opcion: Mode for operating
    :param str enco: encoding

    :return: updated report_dict
    :rtype: dict<python_hashable_type:python_type>
    '''
    modo = glod.get_append_mode()
    file_to_operate = glod.get_empty_string()
    for ruta_fichero in array_rutas_ficheros:
        if opcion == 0:
            modo = glod.get_write_mode()
        if enco != glod.get_none_encoding():
            file_to_operate = codecs.open(ruta_fichero, modo, encoding=enco)
        else:
            file_to_operate = open(ruta_fichero, modo)
        file_to_operate.write(mensaje)
        file_to_operate.close()


def register_target_values_distribution(diccionario_targets_valores, mensaje,
                                        array_rutas_ficheros_log, report_dict, enco, opcion=glod.get_empty_string()):
    '''
    This function allows to register in the report_dict the distribution
    of the targets for the current execution

    Parameters:
    :param pandas_dataframe df_datos: dataframe with all the available data for the current event
    :param target: Current objective feature
    :param str mensaje: Message to register in the log
    :param list array_rutas_ficheros_log: List with the path to the log registers
    :param dict report_dict: Dictionary with the current processed information about the event

    :return: updated report_dict
    :rtype: dict<python_hashable_type:python_type>
    '''

    register_log(array_rutas_ficheros_log, mensaje, glod.get_empty_string(), enco)
    for key in sorted(diccionario_targets_valores):
        register_log(array_rutas_ficheros_log, "\ttarget: "+ str(key) +" Number of elements: "+
                     str(diccionario_targets_valores[key]) + "\n", glod.get_empty_string(), enco)
        if opcion == glod.get_empty_string():
            report_dict[glod.get_report_general_info_key()][glod.get_report_generic_target_key()][str(key)] = str(diccionario_targets_valores[key])
        elif opcion == glod.get_train_option_key():
            report_dict = update_train_division(report_dict, str(key),
                                                str(diccionario_targets_valores[key]))
        elif opcion == glod.get_test_option_key():
            report_dict = update_test_division(report_dict, str(key),
                                               str(diccionario_targets_valores[key]))
    return report_dict


def create_basic_report_data_dict(basic_parameters, lista_variables_descartadas, ruta_logo):
    '''This funcion allows to create the structure for the report data dictionary'''

    umbral = basic_parameters[0]
    target = basic_parameters[1]
    main_metric = basic_parameters[2]
    feature_selection_method = basic_parameters[3]
    penalize_falses = basic_parameters[4]

    report_data = {glod.get_title_key(): "Overview With Execution Information",
                   glod.get_logo_key():ruta_logo,
                   glod.get_umbral_key(): str(umbral),
                   glod.get_main_metric_key(): str(main_metric),
                   glod.get_feature_selection_key(): str(feature_selection_method),
                   glod.get_penalization_name(): str(penalize_falses),
                   glod.get_objective_target_key(): target,
                   glod.get_variables_key():{glod.get_deleted_by_user_key():\
                                         lista_variables_descartadas},
                   glod.get_general_info_execution_key():glod.get_empty_string()
                  }
    return report_data

def create_report_data_dict(evento, umbral, target, lista_variables_descartadas, ruta_logo):
    '''This funcion allows to create the structure for the report data dictionary
    for the current event'''

    report_data = {glod.get_objective_target_key(): target,
                   glod.get_event_key():evento,
                   glod.get_logo_key():ruta_logo,
                   glod.get_report_general_info_key():{glod.get_report_generic_target_key():{},
                                                       glod.get_variables_key():{glod.get_deleted_by_user_key():lista_variables_descartadas, glod.get_empty_or_constant_key():[], glod.get_score_relevant_key():[]},
                                                       glod.get_training_division_key():{},
                                                       glod.get_test_division_key():{},
                                                      },
                   glod.get_umbral_key(): str(umbral),
                   glod.get_warning_key(): glod.get_empty_string()
                  }
    return report_data

def update_report_warning_info(report_dict, informacion):
    '''This funcion allows to register warning info in the report'''

    report_dict[glod.get_warning_key()] = informacion
    return report_dict

def update_report_percentil(report_dict, valor):
    '''This funcion allows to register the percentil value in the report'''

    report_dict[glod.get_umbral_key()] = valor
    return report_dict

def update_report_empty_constant_features(report_dict, lista_vacias_constantes):
    '''This funcion allows to register the empty or constant features in the report'''

    report_dict[glod.get_report_general_info_key()][glod.get_variables_key()]\
    [glod.get_empty_or_constant_key()] = lista_vacias_constantes
    return report_dict

def update_report_relevant_user_features(report_dict, lista_importantes):
    '''This funcion allows to register the relevant features in the report'''

    report_dict[glod.get_report_general_info_key()][glod.get_variables_key()]\
    [glod.get_user_requested_key()] = lista_importantes
    return report_dict

def update_report_user_discarded_features(report_dict, lista_descartadas):
    '''This funcion allows to register the discarded features in the report'''

    report_dict[glod.get_report_general_info_key()][glod.get_variables_key()]\
    [glod.get_user_discarded_key()] = lista_descartadas
    return report_dict

def update_report_training_models_features(report_dict, diccionario_variables_scores):
    '''This funcion allows to register features that are used to train the models
    in the report'''

    report_dict[glod.get_report_general_info_key()][glod.get_variables_key()]\
    [glod.get_score_relevant_key()] = diccionario_variables_scores
    return report_dict

def update_report_full_list_features_used_in_process(report_dict, lista_variables_in_process):
    '''This funcion allows to register features the full list of features that could be used
    to train the models in the report'''

    report_dict[glod.get_report_general_info_key()][glod.get_variables_key()]\
    [glod.get_used_in_process()] = lista_variables_in_process
    return report_dict

def update_train_division(report_dict, key, valor):
    '''This funcion allows to register the distribution of observations that will be used
    to train the models in the report'''

    report_dict[glod.get_report_general_info_key()][glod.get_training_division_key()][key] = valor
    return report_dict

def update_test_division(report_dict, key, valor):
    '''This funcion allows to register the distribution of observations that will be used
    to test the models in the report'''

    report_dict[glod.get_report_general_info_key()][glod.get_test_division_key()][key] = valor
    return report_dict

def add_model_to_report(report_dict, modelo):
    '''This funcion allows to register the information about the trained model in the report'''

    report_dict[modelo] = {glod.get_parameters_key():{},
                           glod.get_time_parameters_key():{},
                           glod.get_accuracy_parameter_name():0,
                           glod.get_decision_index_key():0,
                          }
    return report_dict

def update_model_parameters(report_dict, modelo, parametro, valor):
    '''This funcion allows to register the information about the parameters of the
    trained model in the report'''

    report_dict[modelo][glod.get_parameters_key()][parametro] = valor
    return report_dict

def update_report_model_time(report_dict, modelo, parametro_temporal, valor):
    '''This funcion allows to register the information about the elapsed time to create
    trained model in the report'''

    report_dict[modelo][glod.get_time_parameters_key()][parametro_temporal] = valor
    return report_dict

def actualizar_accuracy_modelo(report_dict, modelo, valor):
    '''This funcion allows to register the information about the accuracy for
    trained model in the report'''

    report_dict[modelo][glod.get_accuracy_parameter_name()] = valor
    return report_dict

def update_model_feature(report_dict, modelo, parametro, valor):
    '''This funcion allows to register the information about a generic feature of
    trained model in the report'''

    report_dict[modelo][parametro] = "'"+ valor+"'"
    return report_dict

def update_report_current_model_decision_index(report_dict, modelo, valor):
    '''This funcion allows to register the information about the overall score of the
    trained model in the report'''

    report_dict[modelo][glod.get_decision_index_key()] = valor
    return report_dict

def get_string_with_ranking_of_models(lista_modelo_ranking, modelo_actual):
    '''This funcion allows to get the ranking of models in a printable form'''

    informacion = "<h3>&nbsp;Models ranking</h3><p>"
    for par_modelo_indice in lista_modelo_ranking:
        modelo = par_modelo_indice[0]
        indice_dec = par_modelo_indice[1]
        if modelo_actual == modelo:
            informacion += "&nbsp;&nbsp;<strong>"+modelo+":&nbsp;&nbsp;"+\
            str(float(round(indice_dec, 4))) +"</strong></br>"
        else:
            informacion += "&nbsp;&nbsp;" + modelo + ":&nbsp;&nbsp;" +\
            str(float(round(indice_dec, 4))) + "</br>"
    informacion += "</p>"

    return informacion


def create_report_current_execution(report_dict, basic_lists, diccionario_aprendizajes,
                                    ruta_relativa_datos_auxiliares, ruta_directorio_resultados):
    '''This funcion allows to create a pdf with the information about the current
    process that is going to take place'''

    env = Environment(loader=FileSystemLoader('.'))
    ruta_plantilla_temporal = os.path.join(ruta_relativa_datos_auxiliares, 'temp_html.html')
    template = env.get_template(ruta_relativa_datos_auxiliares + '/' +\
                               glod.get_general_execution_template_name())

    template_vars = {glod.get_title_key(): report_dict[glod.get_title_key()],
                     glod.get_logo_key():encode_image(report_dict[glod.get_logo_key()].replace('\'', glod.get_empty_string())),
                     glod.get_general_info_execution_key():glod.get_empty_string()
                    }

    lista_eventos = basic_lists[0]
    lista_variables_usuario = basic_lists[1]
    lista_listas_variables_descartadas = basic_lists[2]
    lista_aprendizajes = basic_lists[3]
    lista_modelos = basic_lists[4]
    #General parameters (target,umbral,variables_descartadas)
    target = report_dict[glod.get_objective_target_key()]
    umbral = report_dict[glod.get_umbral_key()]
    main_metric = report_dict[glod.get_main_metric_key()]
    feature_selection_method = report_dict[glod.get_feature_selection_key()]
    penalize_falses = report_dict[glod.get_penalization_name()]
    lista_variables_descartadas = report_dict[glod.get_variables_key()]\
    [glod.get_deleted_by_user_key()]

    tabulacion = "&nbsp;&nbsp;&nbsp;&nbsp;"
    informacion = "<h3>Common Parameters </h3></p>"
    informacion += tabulacion+tabulacion + "<i>Objective Target: </i>" + target + "</br></br>"
    informacion += tabulacion+tabulacion + "<i>Percentil for Scoring Function: </i>" + umbral +\
    "</br></br>"
    informacion += tabulacion+tabulacion + "<i>Main metric: </i>" + main_metric + "</br></br>"
    informacion += tabulacion+tabulacion + "<i>Feature selection method: </i>" + \
    feature_selection_method + "</br></br>"
    informacion += tabulacion+tabulacion + "<i>Penalize falses: </i>" + penalize_falses +\
    "</br></br>"
    informacion += tabulacion+tabulacion + "<i>Common Discarded Variables:</i></br>"
    for variable_descartada in lista_variables_descartadas:
        informacion += tabulacion+tabulacion+tabulacion + variable_descartada + "</br>"
    if lista_variables_descartadas == []:
        informacion += tabulacion+"No variables were selected to be discarded</br>"
    informacion += "</p>"

    informacion += "<h3>Events to be processed: </h3><p>"
    for indice in range(len(lista_eventos)):
        informacion += tabulacion+"<strong>"+ lista_eventos[indice] + "</strong></br>"
        informacion += tabulacion+tabulacion+"<i>Important features for the user:</i> </br>"
        if lista_variables_usuario[indice]:
            for variable in lista_variables_usuario[indice]:
                informacion += tabulacion+tabulacion+tabulacion+variable + "</br>"
        else:
            informacion += tabulacion+tabulacion+tabulacion + \
            "No important features were specified</br>"
        informacion += "</br>"

        informacion += tabulacion+tabulacion+"<i>Discarded variables by the user:</i> </br>"
        if lista_listas_variables_descartadas[indice]:
            for variable in lista_listas_variables_descartadas[indice]:
                informacion += tabulacion+tabulacion+tabulacion+variable + "</br>"
        else:
            informacion += tabulacion+tabulacion+tabulacion+"No variables were discarded</br>"
        informacion += "</br>"

        informacion += tabulacion+tabulacion+"<i>Learnings to be applied: </i></br>"
        aprendizaje = lista_aprendizajes[indice]
        modelos = lista_modelos[indice]
        if aprendizaje == glod.get_all_learning_modes_name():#looping supervised models
            informacion += tabulacion+tabulacion+tabulacion+"<u>" +\
            str(diccionario_aprendizajes[1]) + "</u>:</br>"
            modelos_sup = modelos[0]
            for modelo_act in modelos_sup:
                informacion += tabulacion+tabulacion+tabulacion+tabulacion + modelo_act + "</br>"
            informacion += "</br>"

        else:
            informacion += tabulacion+tabulacion+tabulacion+"<u>"+aprendizaje + "</u>:</br>"
            for modelo_act in modelos:
                informacion += tabulacion+tabulacion+tabulacion+tabulacion + modelo_act + "</br>"

        informacion += "</p>"

        template_vars[glod.get_general_info_execution_key()] = informacion

    with codecs.open(ruta_plantilla_temporal, glod.get_write_mode()) as output_file:
        output_file.write(template.render(template_vars))


    with codecs.open(ruta_plantilla_temporal, glod.get_read_mode()) as html_leido:
        pdf_resultante = os.path.join(ruta_directorio_resultados,
                                      "General_execution_report_"+ target +".pdf")
        with open(pdf_resultante, glod.get_writebyte_mode()) as gen_report:
            pisa.CreatePDF(html_leido.read(), gen_report)
            logging.getLogger("xhtml2pdf").addHandler(PisaNullHandler())

    if os.path.exists(ruta_plantilla_temporal):
        os.remove(ruta_plantilla_temporal)


def create_report_current_model(report_dict, lista_modelos, ruta_relativa_datos_auxiliares,
                                ruta_directorio_informes, enco):
    '''This funcion allows to get information of the current model in pdf format
    with the full charactristics fo the model'''

    env = Environment(loader=FileSystemLoader('.'))
    ruta_plantilla_temporal = os.path.join(ruta_relativa_datos_auxiliares, 'temp_html.html')

    if lista_modelos == []: #if process not completed
        template = env.get_template(ruta_relativa_datos_auxiliares + '/' +\
                                    glod.get_incomplete_event_report_template_name())

        template_vars = {glod.get_title_key(): "Incomplete Execution Report",
                         glod.get_logo_key(): \
                         encode_image(report_dict[glod.get_logo_key()].replace('\'', glod.get_empty_string())),
                         glod.get_report_generic_target_key(): report_dict[glod.get_objective_target_key()],
                         glod.get_event_key(): report_dict[glod.get_event_key()],
                         glod.get_info_key(): "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" +\
                         report_dict[glod.get_warning_key()]
                        }


        with codecs.open(ruta_plantilla_temporal, glod.get_write_mode(), encoding=enco) as output_file:
            output_file.write(template.render(template_vars))


        with codecs.open(ruta_plantilla_temporal, glod.get_read_mode(), encoding=enco) as html_leido:
            pdf_resultante = os.path.join(ruta_directorio_informes, "report_" +\
                                          report_dict[glod.get_event_key()]+"_incomplete.pdf")
            with open(pdf_resultante, glod.get_writebyte_mode()) as incomplete_rep:
                pisa.CreatePDF(html_leido.read(), incomplete_rep)
                logging.getLogger("xhtml2pdf").addHandler(PisaNullHandler())

    else:
        lista_pares_modelo_indice = auxf.order_models_by_score_and_time(report_dict, lista_modelos)
        template = env.get_template(ruta_relativa_datos_auxiliares + '/' + glod.get_report_template_name())
        for modelo in lista_modelos:
            if modelo in report_dict:

                observations_targets = "<p><strong>Target distribution of observations\
                </strong></br>"
                final_targets_list = list(report_dict[glod.get_report_general_info_key()]\
                                          [glod.get_report_generic_target_key()].keys())
                for ob_target in auxf.natsorted(final_targets_list):
                    observations_targets += "&nbsp;&nbsp;&nbsp;&nbsp;"+ "With target " +\
                    str(ob_target) + " :"+ str(report_dict[glod.get_report_general_info_key()]\
                       [glod.get_report_generic_target_key()][ob_target]) + "</br>"
                observations_targets += "</p>"

                variables_summary = "<p><strong>Summary of variables</strong></br>"
                discarded_for_event = report_dict[glod.get_report_general_info_key()]\
                [glod.get_variables_key()][glod.get_user_discarded_key()]

                variables_summary += "<br><i><u>Deleted by the user at the begining:</i></u></br>"
                for deleted_var in report_dict[glod.get_report_general_info_key()]\
                [glod.get_variables_key()][glod.get_deleted_by_user_key()]:
                    variable_dis = glod.get_empty_string()
                    if deleted_var in discarded_for_event:
                        variable_dis = "<strong>" + deleted_var + "</strong>"
                    else:
                        variable_dis = deleted_var
                    variables_summary += "&nbsp;&nbsp;&nbsp;&nbsp;"+ variable_dis + "</br>"
                variables_summary += "&nbsp;&nbsp;&nbsp;&nbsp;<i>*variables in bold were\
                specified by the user to be discarded specifically for this event<i></br>"
                variables_summary += "</br>"

                variables_summary += "<br><i><u>Deleted in execution time(Empty or Constant)\
                :</i></u></br>"
                for emp_con_var in report_dict[glod.get_report_general_info_key()]\
                [glod.get_variables_key()][glod.get_empty_or_constant_key()]:
                    variables_summary += "&nbsp;&nbsp;&nbsp;&nbsp;"+ emp_con_var + "</br>"
                variables_summary += "</br>"

                variables_summary += "<br><i><u>Requested for the event by the user:</i></u></br>"
                for req_var in report_dict[glod.get_report_general_info_key()]\
                [glod.get_variables_key()][glod.get_user_requested_key()]:
                    variables_summary += "&nbsp;&nbsp;&nbsp;&nbsp;"+ req_var + "</br>"
                variables_summary += "</br>"

                variables_summary += "<br><i><u>Used during the process:</i></u></br>"

                diccionario_relevantes_mif = report_dict[glod.get_report_general_info_key()]\
                [glod.get_variables_key()][glod.get_score_relevant_key()]
                sorted_relevant_vars = sorted(diccionario_relevantes_mif.items(),
                                              key=operator.itemgetter(1),
                                              reverse=True)
                for relevant_var in sorted_relevant_vars:
                    rel_variable = relevant_var[0]
                    rel_variable = "<strong>" + rel_variable +'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'+\
                    str(diccionario_relevantes_mif[rel_variable]) +"</strong>"
                    variables_summary += "&nbsp;&nbsp;&nbsp;&nbsp;"+ rel_variable + "</br>"

                for relevant_var in report_dict[glod.get_report_general_info_key()][glod.get_variables_key()][glod.get_used_in_process()]:
                    if relevant_var not in diccionario_relevantes_mif:
                        variables_summary += "&nbsp;&nbsp;&nbsp;&nbsp;"+ relevant_var + "</br>"
                variables_summary += "&nbsp;&nbsp;&nbsp;&nbsp;<i>*variables in bold were used\
                to train the models<i></br>"
                variables_summary += "</p>"


                #Information about the model
                accuracy = "</br></br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\
                &nbsp;<strong>Accuracy: "+\
                str(float(round(report_dict[modelo][glod.get_accuracy_parameter_name()], 5)))+\
                "</strong>"

                ranking = get_string_with_ranking_of_models(lista_pares_modelo_indice, modelo)

                model_info = "<p><strong>Parameters used to configure the model</strong></br>"
                for param in report_dict[modelo][glod.get_parameters_key()]:
                    model_info += "&nbsp;&nbsp;&nbsp;&nbsp;<i>"+ param + "</i>: " +\
                    str(report_dict[modelo][glod.get_parameters_key()][param]) + "</br>"
                model_info += "</p>"

                time_info = "<p><strong>Time elapsed</strong></br>"
                tiempo_seleccion_parametros = report_dict[modelo][glod.get_time_parameters_key()][glod.get_time_sel_finish_key()] - report_dict[modelo][glod.get_time_parameters_key()][glod.get_time_sel_init_key()]
                tiempo_entrenamiento = report_dict[modelo][glod.get_time_parameters_key()][glod.get_time_train_finish_key()] - report_dict[modelo][glod.get_time_parameters_key()][glod.get_time_train_init_key()]
                time_info += "&nbsp;&nbsp;&nbsp;&nbsp;"+ "Parameters selection time: "+\
                str(tiempo_seleccion_parametros) + "</br>"
                time_info += "&nbsp;&nbsp;&nbsp;&nbsp;"+ "Training time: "+\
                str(tiempo_entrenamiento) + "</br>"
                time_info += "</p>"


                total_train = 0.0
                vector_of_targets = []
                vector_of_values_by_target = []
                vector_of_percentages_by_target = []
                train_distribution_info = "<p></br><strong>Training Data Distribution\
                </strong></br>"
                for train_target in auxf.natsorted(list(report_dict[glod.get_report_general_info_key()][glod.get_training_division_key()].keys())):
                    train_distribution_info += "&nbsp;&nbsp;&nbsp;&nbsp;"+ "With target " + str(train_target) + " :"+ str(report_dict[glod.get_report_general_info_key()][glod.get_training_division_key()][train_target]) + "</br>"
                    vector_of_targets.append(train_target)
                    vector_of_values_by_target.append(float(report_dict[glod.get_report_general_info_key()][glod.get_training_division_key()][train_target]))
                    total_train += float(report_dict[glod.get_report_general_info_key()][glod.get_training_division_key()][train_target])
                train_distribution_info += "</p>"
                #getting null train accuracy
                null_train_accuracy = 0.0
                for indice_t in range(len(vector_of_values_by_target)):
                    vector_of_percentages_by_target.append(round(vector_of_values_by_target[indice_t]/total_train, 4))

                null_train_accuracy = max(vector_of_percentages_by_target)

                total_test = 0.0
                vector_of_targets = []
                vector_of_values_by_target = []
                vector_of_percentages_by_target = []
                test_distribution_info = "<p><strong>Test Data Distribution</strong></br>"
                for test_target in auxf.natsorted(list(report_dict[glod.get_report_general_info_key()][glod.get_test_division_key()].keys())):
                    test_distribution_info += "&nbsp;&nbsp;&nbsp;&nbsp;"+ "With target " + str(test_target) + " :"+ str(report_dict[glod.get_report_general_info_key()][glod.get_test_division_key()][test_target]) + "</br>"
                    vector_of_targets.append(test_target)
                    vector_of_values_by_target.append(float(report_dict[glod.get_report_general_info_key()][glod.get_test_division_key()][test_target]))
                    total_test += float(report_dict[glod.get_report_general_info_key()][glod.get_test_division_key()][test_target])
                test_distribution_info += "</p>"
                null_test_accuracy = 0.0
                for indice_t in range(len(vector_of_values_by_target)):
                    vector_of_percentages_by_target.append(round(vector_of_values_by_target[indice_t]/total_test, 4))
                null_test_accuracy = max(vector_of_percentages_by_target)

                event = report_dict[glod.get_event_key()]
                template_vars = {glod.get_title_key(): "Execution Report",
                                 glod.get_logo_key():encode_image(report_dict[glod.get_logo_key()].replace('\'', glod.get_empty_string())),
                                 glod.get_model_key(): modelo,
                                 glod.get_report_generic_target_key():\
                                 report_dict[glod.get_objective_target_key()],
                                 glod.get_event_key(): event,
                                 glod.get_accuracy_parameter_name():\
                                 str(accuracy)+"<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\
                                 &nbsp;&nbsp;&nbsp;&nbsp;<strong>Null train acc: "+\
                                 str(null_train_accuracy)+"</strong>"+"<br>&nbsp;&nbsp;&nbsp;\
                                 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\
                                 <strong>Null test acc: "+ str(null_test_accuracy)+\
                                 "</strong></p>",
                                 glod.get_models_ranking_key(): ranking,
                                 glod.get_observations_targets_key(): observations_targets,
                                 glod.get_variables_summary_key(): variables_summary,
                                 glod.get_models_info_key(): model_info,
                                 glod.get_time_info_key(): time_info,
                                 glod.get_train_distribution_info_key(): train_distribution_info,
                                 glod.get_test_distribution_info_key(): test_distribution_info
                                }
                template_vars[glod.get_metrics_info_key()] = glod.get_empty_string()
                for metric in report_dict[modelo][glod.get_metrics_micro_avg_key()]:
                    template_vars[glod.get_metrics_info_key()] += "<p>"+"<strong>"+metric+"</strong>: " + report_dict[modelo][glod.get_metrics_micro_avg_key()][metric] +"</br>"
                template_vars[glod.get_metrics_info_key()] += "</p>"

                if glod.get_model_parameters_plot_name() in report_dict[modelo]:
                    template_vars[glod.get_image_parameters_accuracy_key()] = encode_image(report_dict[modelo][glod.get_model_parameters_plot_name()].replace('\'', glod.get_empty_string()))

                if glod.get_confussion_matrix_train_path_key() in report_dict[modelo]:
                    template_vars[glod.get_conf_train_img_key()] = encode_image(report_dict[modelo][glod.get_confussion_matrix_train_path_key()].replace('\'', glod.get_empty_string()))

                if glod.get_confussion_matrix_test_path_key() in report_dict[modelo]:
                    template_vars[glod.get_conf_test_img_key()] = encode_image(report_dict[modelo][glod.get_confussion_matrix_test_path_key()].replace('\'', glod.get_empty_string()))

                if glod.get_learning_curve_key() in report_dict[modelo]:
                    template_vars[glod.get_learning_curve_key()] = encode_image(report_dict[modelo][glod.get_learning_curve_key()].replace('\'', glod.get_empty_string()))


                metrics_by_label = "<table width='100%' border='1' cellspacing='0' cellpadding='5'>"
                keys = glod.get_empty_string()
                for elemento in auxf.natsorted(list(report_dict[modelo][glod.get_metrics_key()].keys())):
                    if keys == glod.get_empty_string():
                        keys = report_dict[modelo][glod.get_metrics_key()][elemento].keys()
                        metrics_by_label += "<tr><td align='center' class='black'>"+ glod.get_report_generic_target_key() +"</td>"
                        for cabecera in keys:
                            metrics_by_label += "<td align='center' class='black'>" + cabecera +"</td>"
                        metrics_by_label += "</tr>"
                    metrics_by_label += "<tr><td>" + elemento.replace('target_', glod.get_empty_string()) + "</td>"
                    for key in keys:
                        metrics_by_label += "<td>"+str(report_dict[modelo][glod.get_metrics_key()][elemento][key])+"</td>"
                    metrics_by_label += "</tr>"
                metrics_by_label += "</table>"
                template_vars[glod.get_metrics_by_label_key()] = metrics_by_label

                #generamos el html
                with codecs.open(ruta_plantilla_temporal, glod.get_write_mode(), encoding=enco) as output_file:
                    output_file.write(template.render(template_vars))

                #generamos el pdf
                with codecs.open(ruta_plantilla_temporal, mode=glod.get_read_mode(), encoding=enco) as read_html:
                    pdf_resultante = os.path.join(ruta_directorio_informes, modelo + "_report_for_"+ event +".pdf")
                    with open(pdf_resultante, mode=glod.get_writebyte_mode()) as pdf_gen:
                        pisa.CreatePDF(read_html.read(), pdf_gen)
                        logging.getLogger("xhtml2pdf").addHandler(PisaNullHandler())

    if os.path.exists(ruta_plantilla_temporal):
        os.remove(ruta_plantilla_temporal)

def create_report_prediction(report_dict, event_target, ruta_relativa_datos_auxiliares,
                             ruta_directorio_informes, enco):
    '''This funcion allows to get the pdf for the current model with the information
    obtained fate rthe prediction phase'''

    env = Environment(loader=FileSystemLoader('.'))
    ruta_plantilla_temporal = os.path.join(ruta_relativa_datos_auxiliares, 'temp_html.html')

    template = env.get_template(ruta_relativa_datos_auxiliares + '/' + glod.get_prediction_template_name())

    event = event_target[0]
    target_to_predict = event_target[1]

    dic_info_event = report_dict[event]
    summary_target_to_predict = glod.get_empty_string()

    template_vars = {glod.get_title_key(): "Prediction report for " + event,
                     glod.get_logo_key():\
                     encode_image(report_dict[glod.get_logo_key()].replace('\'', glod.get_empty_string())),
                    }

    if target_to_predict in dic_info_event:
        model = str(dic_info_event[target_to_predict][glod.get_best_model_key()])
        model = model.split("(")
        model = model[0]
        summary_target_to_predict = "<p><strong>Target: <strong>" + '&nbsp' +\
        target_to_predict + "</br></br>"
        summary_target_to_predict += "<p><strong>Model: <strong>" + '&nbsp' + model + "</br>"
        summary_target_to_predict += "<p><strong>Accuracy: <strong>" + '&nbsp' +\
        str(dic_info_event[target_to_predict][glod.get_accuracy_parameter_name()]) + "</br>"
        summary_target_to_predict += "<strong>Correct classifications: <strong>" + '&nbsp' +\
        str(dic_info_event[target_to_predict]['Correct']) + "</br>"
        summary_target_to_predict += "<strong>Total number of observations: <strong>" + '&nbsp' +\
        str(dic_info_event[target_to_predict]['Total']) + "</br>"
        summary_target_to_predict += "<strong>Total number of unknown observations classified: <strong>" + '&nbsp' + str(dic_info_event[target_to_predict]['Predicted']) + "</br>"
        cm_target = encode_image(dic_info_event[target_to_predict]\
                                 ['target_to_predict_cm'].replace('\'', glod.get_empty_string()))
        template_vars['target_to_predict_cm'] = cm_target

    template_vars['target'] = summary_target_to_predict


    with codecs.open(ruta_plantilla_temporal, glod.get_write_mode(), encoding=enco) as output_file:
        output_file.write(template.render(template_vars))

    with codecs.open(ruta_plantilla_temporal, mode=glod.get_read_mode(), encoding=enco) as read_html:
        pdf_resultante = os.path.join(ruta_directorio_informes,\
                                    "Prediction_report_for_"+ event +".pdf")
        with open(pdf_resultante, mode=glod.get_writebyte_mode()) as pdf_gen:
            pisa.CreatePDF(read_html.read(), pdf_gen)
            logging.getLogger("xhtml2pdf").addHandler(PisaNullHandler())

    if os.path.exists(ruta_plantilla_temporal):
        os.remove(ruta_plantilla_temporal)


def create_report_current_dictionary_models(dictionary_of_models, basic_paths,
                                            list_of_parameters_models_events_dict, logo_path, enco):
    '''This funcion allows to get the pdf file with the current status of the models,
    relevant features and the events to which are applied'''

    ruta_relativa_datos_auxiliares = basic_paths[0]
    ruta_directorio_resultados = basic_paths[1]
    env = Environment(loader=FileSystemLoader('.'))
    ruta_plantilla_temporal = os.path.join(ruta_relativa_datos_auxiliares, 'temp_html.html')
    template = env.get_template(ruta_relativa_datos_auxiliares + '/' +\
                                glod.get_dictionary_models_template_name())

    tabulacion = "&nbsp;&nbsp;&nbsp;&nbsp;"

    template_vars = {glod.get_title_key(): "Report of the information of the Dictionary of models",
                     glod.get_logo_key(): encode_image(logo_path.replace('\'', glod.get_empty_string()))
                    }

    list_elements = [list_of_parameters_models_events_dict[0],
                     list_of_parameters_models_events_dict[3],
                     list_of_parameters_models_events_dict[1]]
    informacion = glod.get_empty_string()
    for event in dictionary_of_models:
        informacion += "<strong><u>"+ event +"</u></strong></br></br>"
        for target in dictionary_of_models[event]:
            informacion += tabulacion + tabulacion + "<strong><i>Target:</i></strong>" + "&nbsp;&nbsp;" + target + "</br>"
            for key in list_elements:
                informacion += tabulacion + tabulacion + "<strong><i>" + key + ": </i></strong>"
                if type(list()) == type(dictionary_of_models[event][target][key]):
                    informacion += "<br>"
                    contador = 0
                    ordered_list_features = sorted(dictionary_of_models[event][target][key])
                    while contador < len(ordered_list_features):
                        element = ordered_list_features[contador]
                        informacion += tabulacion + tabulacion + tabulacion +tabulacion + element + "</br>"
                        contador += 1
                else:
                    informacion += dictionary_of_models[event][target][key] + "</br>"
                    if key == list_of_parameters_models_events_dict[0]:
                        informacion += tabulacion + tabulacion + "<strong><i>best model: </i></strong>&nbsp;&nbsp;" + dictionary_of_models[event][target][list_of_parameters_models_events_dict[1]].split('_')[-1].split('.')[0] + "</br>" #get model name
                        if dictionary_of_models[event][target][key] == glod.get_unsupervised_name():
                            informacion += tabulacion + tabulacion + "<strong><i>dic_reassingment: </i></strong>&nbsp;&nbsp;" + str(dictionary_of_models[event][target][list_of_parameters_models_events_dict[2]]) + "</br>"
            informacion += "</br>"


    if informacion == glod.get_empty_string():
        informacion = "No models were created yet"
    template_vars[glod.get_info_key()] = informacion

    #html
    with codecs.open(ruta_plantilla_temporal, glod.get_write_mode(), encoding=enco) as output_file:
        renderizado = template.render(template_vars)
        output_file.write(renderizado)

    #pdf
    with codecs.open(ruta_plantilla_temporal, mode=glod.get_read_mode(), encoding=enco) as read_html:
        pdf_resultante = os.path.join(ruta_directorio_resultados, "Current_status_dictionary_events_and_models.pdf")
        with open(pdf_resultante, mode=glod.get_writebyte_mode()) as pdf_gen:
            pisa.CreatePDF(read_html.read().encode(enco, 'ignore').decode(enco), pdf_gen)

    if os.path.exists(ruta_plantilla_temporal):
        os.remove(ruta_plantilla_temporal)

    ###################Functions relative to html to generate pdf reports #####################


def create_score_report(directorio_salida, nombre_informe, report, enco):
    '''This funcion allows to get the score fot each relevant feature in
    csv format'''
    try:
        nombre_informe = nombre_informe + '_report_.csv'
        ruta_relativa_informe = os.path.join(directorio_salida, nombre_informe)
        with codecs.open(ruta_relativa_informe, glod.get_write_mode(), encoding=enco) as output_score_report:
            cabecera = u'feature'+u','+u'weight'+u'\n'
            output_score_report.write(cabecera)
            for elemento in report:
                variable_act = elemento[0]
                score = elemento[1]
                output_score_report.write(variable_act  + u',' + str(score) + u'\n')
        mensaje = "Score report created succesfully"
    except Exception as excep:
        print(excep)
        pass

    return mensaje

def save_data_to_file(datos, ruta_directorio, nombre_fichero, delimitador, extension):
    '''This function allows to store data in filex with the specified extension
        datos: data to store
        ruta_directorio: path to the output directory
        nombre_fichero: name of the file that stores the data
        delimitador: delimiter for the fields of the file
        extension: extension for the output file
    '''

    if extension == 'txt':
        nombre_fichero = nombre_fichero + "." + extension
        ruta_destino = os.path.join(ruta_directorio, nombre_fichero)
        np.savetxt(ruta_destino, datos, delimiter=delimitador)
    elif extension == 'csv':
        nombre_fichero = nombre_fichero + "." + extension
        ruta_destino = os.path.join(ruta_directorio, nombre_fichero)
        datos.to_csv(ruta_destino, index=False, sep=delimitador)
    elif extension == 'numpy':
        nombre_fichero = nombre_fichero + "." + 'txt'
        ruta_destino = os.path.join(ruta_directorio, nombre_fichero)
        file_to_write = open(ruta_destino, glod.get_write_mode())
        file_to_write.write("\t Predicted label \n")
        filas = datos.tolist()
        for fila in filas:
            file_to_write.write('\t' + str(fila) + '\n')
        file_to_write.close()


def generate_model_report(porc_acierto_test, porc_acierto_validacion,
                          ruta_directorio_informes_accuracy, nombre_informe):
    '''This function geenrates a report for the prediction phase
    with the information about the performance of the best model'''

    datos_accuracy = np.array([porc_acierto_test, porc_acierto_validacion])
    save_data_to_file(datos_accuracy, ruta_directorio_informes_accuracy, nombre_informe, glod.get_empty_string(), 'txt')
    
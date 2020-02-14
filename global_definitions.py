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

 # pylint: disable = C0111
'''>>>Global definitions<<<'''

def get_config_parser_name():
    return "conf.ini"

def get_encoding():
    return 'utf-8'

def get_logo_name():
    return 'logo.jpg'

def get_output_img_files_extension():
    return 'png'

def get_output_pkl_files_extension():
    return 'pkl'

def get_input_files_extension():
    return 'csv'

def get_log_files_extension():
    return 'log'

def get_empty_string():
    return ''

def get_write_mode():
    return 'w'

def get_writebyte_mode():
    return 'wb'

def get_append_mode():
    return 'a'

def get_read_mode():
    return 'r'

def get_readbyte_mode():
    return 'rb'

'''>>>conf.ini related<<<'''

'''>Logs section'''
def get_log_section_name():
    return  'Logs section'

def get_log_directory_name():
    return  'logs_directory_name'

def get_log_error_name():
    return 'log_error_filename'

def get_log_execution_name():
    return 'execution_log_filename'

def get_log_time_execution_name():
    return 'time_log_filename'

def get_prediction_log_execution_name():
    return 'prediction_log_filename'

def get_prediction_log_time_execution_name():
    return 'prediction_time_log_filename'

'''>Input data section '''
def get_exponent_number():
    return 'exponent_value'

def get_percentage_features():
    return 'n_feat'

def get_input_section_name():
    return 'Input data section'

def get_event_name_feature_name():
    return 'event_name_feature'

def get_non_catalogued_label_name():
    return 'label_non_catalogued'

def get_obsnumber_parameter_name():
    return 'obsnumber'

def get_input_files_delimiter_name():
    return 'input_files_delimiter'

def get_path_to_root_directory_input_files_name():
    return 'path_to_root_directory_input_files'

def get_user_files_directoryname():
    return 'user_files_directoryname'

def get_events_filename_name():
    return 'events_filename'

def get_user_discarded_variables_filename_name():
    return 'user_discarded_variables_filename'

def get_maximum_number_files_to_read_name():
    return 'maximum_number_files_to_read'

def get_maximum_number_observations_to_read_name():
    return 'maximum_number_observations_to_read'

def get_train_test_division_percentaje_name():
    return 'train_test_division_percentaje'

def get_percentage_relevant_variables():
    return 'percentil_relevant_variables'

def get_main_metric_key():
    return 'main_metric'

def get_feature_selection_key():
    return 'feature_selection_method'

def get_default_matrix_name():
    return 'default_matrix'

def get_diagonal_weight_name():
    return 'diagonal_weight'

def get_matrix_of_weights_fp_fn_name():
    return 'matrix_of_weights_fp_fn'

def get_penalization_name():
    return 'penalize_falses'

def get_compute_lc_name():
    return 'compute_lc'

def get_threshold_split():
    return 'threshold_split'

def target_recodified_values():
    return 'target_recodified_values'

'''>Auxiliary section '''
def get_auxiliary_section_name():
    return 'Auxiliary data section'

def get_auxiliary_directory_parameter_name():
    return 'auxiliary_directory_filename'

'''>Output section '''
def get_output_section_name():
    return 'Output data section'

def get_output_files_delimiter_name():
    return 'output_files_delimiter'

def get_output_directory_rootname():
    return 'output_directory_rootname'

def get_output_directory_name_mif_name():
    return 'output_directory_name_mif'

def get_output_directory_name_fisher_name():
    return 'output_directory_name_fisher'

def get_output_directory_name_report_name():
    return 'output_directory_name_report'

def get_output_directory_name_prediction_models():
    return 'output_directory_name_prediction_models'

def get_validation_data_directory_name_name():
    return 'validation_data_directory_name'

def get_prediction_models_dictionary_filename_name():
    return 'prediction_models_dictionary_filename'

'''>Prediction section '''

def get_prediction_section_name():
    return 'Prediction section'

def get_target_parameter_name():
    return 'target_to_predict'

def get_path_to_prediction_models_name():
    return 'path_to_prediction_models_pkl'

def get_delimiter_non_catalogued_data_name():
    return 'non_catalogued_data_csv_separator'

def get_number_of_files_parameter_name():
    return 'number_files_to_catalogue'

def get_name_directory_to_input_files_catalogue():
    return 'path_to_directory_input_files_to_catalogue'


'''>Validation section '''
def get_validation_section_name():
    return 'Validation'

def get_validation_mode_name():
    return 'validation_mode'

def get_validation_division_percentage_name():
    return 'validation_division_percentaje'

'''>>>Python scripts<<<'''
'''>auxiliary_functions.py '''

def get_semi_automated_mode_name():
    return 'semi-automated'

def get_build_files_mode_name():
    return 'build_files'

def get_decision_index_key():
    return 'Decision_index'

def get_time_parameters_key():
    return 'Time_parameters'

def get_time_train_finish_key():
    return 'time_train_finish'

def get_time_train_init_key():
    return 'time_train_init'

def get_current_score_key():
    return 'current_score'

def get_current_time_key():
    return 'current_time'


'''>data_request.py '''
def get_all_learning_modes_name():
    return 'All'

def get_exit_continue_name():
    return 'Exit/Continue'

def get_none_encoding():
    return 'none'

def get_nan_string():
    return 'nan'

def get_relevant_features_input_name():
    return 'relevant_features'

def get_discarded_features_input_name():
    return 'discarded_features'

def get_supervised_name():
    return 'Supervised'

def get_unsupervised_name():
    return 'Unsupervised'

def get_tree_model_name():
    return 'Tree'

def get_ada_model_name():
    return 'Ada'

def get_boosting_model_name():
    return 'Boosting'

def get_random_forest_model_name():
    return 'RandomForest'

def get_mlp_model_name():
    return 'MLPerceptron'

def get_kmeans_model_name():
    return 'Kmeans'


'''>main_prediction.py '''
def get_original_features_key():
    return 'original_features'

def get_current_features_key():
    return 'features'

def get_name_dictionary_know_reco_parameter_name():
    return 'csv_variables'

def get_model_path_key():
    return 'model_path'

def get_best_model_key():
    return 'best_model'

def get_learning_key():
    return 'learning'

def get_reasignment_dict_key():
    return 'dict_reassignment'

def get_path_predicted_data_key():
    return 'path_to_predicted_data'


'''>main_train.py '''
def get_user_discarded_common_features():
    return 'discarded_common_features'

def get_generated_data_directory_name():
    return 'generated_data'

def get_train_option_key():
    return 'Train'

def get_test_option_key():
    return 'Test'

def get_time_model_init_key():
    return 'time_model_init'

def get_time_sel_init_key():
    return 'time_sel_init'

def get_time_sel_finish_key():
    return 'time_sel_finish'

def get_info_key():
    return 'Info'

def get_confusion_matrix_incorrect_classifications_unsu_name():
    return 'confusion_matrix_incorrect_classifications'

def get_confusion_matrix_incorrect_classifications_train_name():
    return 'confusion_matrix_incorrect_classifications_train'

def get_confusion_matrix_incorrect_classifications_test():
    return 'confusion_matrix_incorrect_classifications_test'

def get_mcc_metric_key():
    return 'mcc'

def get_confussion_matrix_train_path_key():
    return 'Confusion_matrix_train_path'

def get_confussion_matrix_test_path_key():
    return 'Confusion_matrix_test_path'



'''>metrics.py '''
def generic_target_dict_key():
    return 'target_'

def get_metrics_key():
    return 'metrics'

def get_metrics_by_label_key():
    return 'metrics_by_label'

def get_metrics_macro_avg_key():
    return 'metrics_macro_avg'

def get_metrics_micro_avg_key():
    return 'metrics_micro_avg'

def get_precision_key():
    return 'precision'

def get_recall_key():
    return 'recall'

def get_specificity_key():
    return 'specificity'

def get_f1_score_key():
    return 'f1_score'

def get_mcc_key():
    return 'mcc'

def get_true_label_text():
    return 'True label'

def get_predicted_label_text():
    return 'Predicted label'

def get_training_sample_text():
    return 'Training sample'

def get_success_rate_text():
    return 'Training sample'

def get_success_rate_lengend_text():
    return 'succes rate in-sample'

def get_cross_validation_rate_legend_text():
    return 'cross-validation rate'

'''>parameters_selection.py '''
def get_model_max_depth_parameter_name():
    return 'max_depth'

def get_model_n_estimators_parameter_name():
    return 'n_estimators'

def get_model_hidden_layer_sizes():
    return 'hidden_layer_sizes'

def get_model_hidden_layer_1_len_name():
    return 'length layer 1'

def get_model_hidden_layer_2_len_name():
    return 'length layer 2'

def get_number_of_groups_parameter_name():
    return 'number of groups'

def get_number_of_initializations_parameter_name():
    return 'number of initializations'

def get_model_parameters_plot_name():
    return 'parameters_plot'

def get_accuracy_parameter_name():
    return 'Accuracy'


'''>>>Report definitions<<<'''
''' header '''
def get_title_key():
    return 'title'

def get_logo_key():
    return 'logo'

def get_model_key():
    return 'model'

def get_event_key():
    return 'event'

def get_report_generic_target_key():
    return 'target'

def get_umbral_key():
    return 'Umbral'

def get_warning_key():
    return 'Warning_info'

def get_objective_target_key():
    return 'Objective_target'

def get_variables_key():
    return 'Variables'

def get_deleted_by_user_key():
    return 'Deleted_by_user'

def get_empty_or_constant_key():
    return 'Empty_constant'

def get_user_requested_key():
    return 'User_requested'

def get_user_discarded_key():
    return 'User_discarded'

def get_score_relevant_key():
    return 'score_relevant'

def get_used_in_process():
    return 'Used_in_process'

def get_training_division_key():
    return 'Training_division'

def get_test_division_key():
    return 'Test_division'

def get_parameters_key():
    return 'Parameters'

def get_general_info_execution_key():
    return 'General_information_execution'

def get_report_general_info_key():
    return 'General_info'

def get_image_parameters_accuracy_key():
    return 'Image_parameters_accuracy'

def get_variables_summary_key():
    return 'Variables_summary'

def get_metrics_info_key():
    return 'Metrics_info'

def get_conf_train_img_key():
    return 'Conf_train_img'

def get_conf_test_img_key():
    return 'Conf_test_img'

def get_learning_curve_key():
    return 'lc_0_0'

def get_general_execution_template_name():
    return 'general_execution_template.html'

def get_incomplete_event_report_template_name():
    return 'incomplete_event_report_template.html'

def get_report_template_name():
    return 'report_template.html'

def get_prediction_template_name():
    return 'prediction_template.html'

def get_dictionary_models_template_name():
    return 'dictionary_models_template.html'


def get_models_ranking_key():
    return  'Models_ranking'

def get_observations_targets_key():
    return 'observations_targets'

def get_models_info_key():
    return 'Model_info'

def get_time_info_key():
    return 'Time_info'

def get_train_distribution_info_key():
    return 'Train_distribution_info'

def get_test_distribution_info_key():
    return 'Test_distribution_info'
    
# Print_freq: print results (accuracy) every Print_freq iterations or epochs during running the code
# Save_dir:  All the relevant results, such as trained model, corresponding parameters, metrics of results: accuracy, curves, etc. The results will be saved in a file and will be used in the future for plotting and interpretation
# GPUs:   a list of GPU ids that are used in parallel. 0-9 corresponding to our gpu id. 10 means use several gpus in parallel (MSDNet use several GPU in parallel)
# Dataset_root: The directory where all the datasets are saved.
# Backbone: the backbone NN network: 'vgg','resnet','wideresnet','mobilenet'
# datasets:   "CIFAR10" "CIFAR100" "TinyImageNet"
# models: "SDN", "MSDNet", "l2stop"
# Log_file: This is the log file where all the metric results of each model are stored, such as accuracy and running time.
# Add_ic_vgg: which intermediate layers will be selected to attach the internal classifiers. 1 means there is one internal classifier in the corresponding intermediate layer.
# Add_ic_resnet: same as above.
# Add_ic_wideresnet: same as above.
# Add_ic_mobilenet: same as above.
# Model-pretrain: We will use pretrained models for all the backbone networks. The source of pretrained models can be found from the related papers published at ICLR.
# 1: pretrained model from the internet, some method might not be competitive because of the ready pretrained model.
# 2 : resnet-56 used in sdn approach. There is no ready pretrained resnet-56 model.
# https://pocketflow.github.io/pre_trained_models/

# we need a small and simple dataset that is used to test if the models are working reasonable. that means, the accuracies of the models on this simple dataset will be all high.
# 2 epochs
# mnist
# the approximate running times for the three models: SDN (15 mins, 5 epochs), MSDNet (<=1 hours 2 epochs), l2stop (< 15 min 5 epochs)
# the approximate running times for the three models: SDN (1 hour, 20 epochs), MSDNet (2 hours 5 epochs), l2stop (1 hour 20 epochs), BranchNet (5 CPU hours, 50 epochs)


# the approximate running times for the three models: SDN (15 hour, 300 epochs), MSDNet (120 hours 300 epochs), l2stop (15 hour 300 epochs)

# approximated running time:
# CIFAR-10:     SDN (4-5 hours 100 epochs), MSDNet (>120 hours 300 epochs), l2stop (4-5 hours 100 epochs)
# CIFAR-100:    SDN (5-6 hours 100 epochs), MSDNet (>120 hours 300 epochs), l2stop (5-6 hours 100 epochs)
# TinyImageNet: SDN (8-9 hours 100 epochs), MSDNet (>150 hours 300 epochs), l2stop (8-9 hours 100 epochs)

# setting 1: It is only used to test if the models will run without reporting error




# General_Parameters = {
# "save_dir"          : "saved_results",
# "log_file"          : "log_file_name.log",
# "seed"              : 0,
# "GPUs"              : [1],
# "print_freq"        : 100,
# "dataset"           : ["mnist"],
# "dataset_root"      : "/home/cxl173430/data",
# "models"            : ["SDN", "MSDNet", "l2stop"],
# "GPUs"              : [[1], [2, 3], [4]],
# "backbone"          : ["vgg"],
# "mini_batch_size"   : 64,
# }
#
# SDN = {
# "epochs"            : 100,
# "learning_rate"     : 0.1,
# "optimizer"         : "adam",
# "add_ic_vgg"        : [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
# "add_ic_resnet"     : [[0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 0]],
# "add_ic_wideresnet" : [[0, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 0]],
# "add_ic_mobilenet"  : [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
# }
#
# MSDNet = {
# "epochs"            : 123,
# "learning_rate"     : 0.05,
# "optimizer"         : "adam",
# }
#
# l2stop = {
# "epochs"            : 100,
# "learning_rate"     : 0.05,
# "optimizer"         : "adam",
# }

General_Parameters = {
"save_dir"          : "saved_results_test_0712_3",
"log_file"          : "log_file_name_0708_ffml_test.log",
"seed"              : 0,
# "print_freq"        : 100,
"dataset"           : ["ny_stop_and_frisk"],
"dataset_root"      : "/home/feng/FFML/data",
# "models"            : ["l2stop", "SDN", "MSDNet"],
"model"             : None,
# "GPUs"              : [[0]],
# "backbone"          : ["vgg"],
# "mini_batch_size"   : 64,
}

# SDN = {
# "log_file"          : "log_file_name_0708_SDN_test_1.log",
# "epochs"            : 1,
# "learning_rate"     : 0.1,
# "optimizer"         : "adam",
# "add_ic_vgg"        : [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
# "add_ic_resnet"     : [[0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 0]],
# "add_ic_wideresnet" : [[0, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 0]],
# "add_ic_mobilenet"  : [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
# }
#
# MSDNet = {
# "log_file"          : "log_file_name_0708_MSDNet_test_1.log",
# "epochs"            : 1,
# "learning_rate"     : 0.05,
# "optimizer"         : "adam",
# }
#
# l2stop = {
# "log_file"          : "log_file_name_0708_l2stop_test_1.log",
# "epochs"            : 1,
# "epochs_policy_net" : 2,
# "learning_rate"     : 0.05,
# "optimizer"         : "adam",
# "add_ic_vgg"        : [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
# "add_ic_resnet"     : [[0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 0]],
# "add_ic_wideresnet" : [[0, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 0]],
# "add_ic_mobilenet"  : [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
# }

masked_ftml= {
"log_file"          : "log_file_name_0708_ffml_test.log",
"val_batch_size" : 0.9,  # data points for validation
"K" : 100,  # few shots in support
"Kq" : 200,  # shots for query kq= k*2
"inner_steps" : 1,  # gradient steps in the inner loop
"num_iterations" : 100,  # outer iteration
"meta_batch" : 32,
"eta_1" : 0.00001,  # step size of inner primal update
"eta_3" : 0.00001,  # step size of outer primal update
"num_neighbors" : 3,
"d_feature" : 51  # feature size of the data set cls_syn_data:2; adult:16; communities_and_crime:100; bank:16; census_income:36
}

twp= {
"log_file"          : "log_file_name_0708_ffml_test.log",
"val_batch_size" : 0.9,  # data points for validation
"K" : 100,  # few shots in support
"Kq" : 200,  # shots for query kq= k*2
"inner_steps" : 1,  # gradient steps in the inner loop
"num_iterations" : 100,  # outer iteration
"meta_batch" : 32,
"eta_1" : 0.00001,  # step size of inner primal update
"eps" : 0.45,  # step size of outer primal update
"num_neighbors" : 3,
"d_feature" : 51  # feature size of the data set cls_syn_data:2; adult:16; communities_and_crime:100; bank:16; census_income:36
}

import json
import time
import os
import sys
import numpy as np
import pickle
import json
from collections import defaultdict
from sklearn.metrics import classification_report

import config

import torch
import torch.nn as nn
import argparse

united_parser = argparse.ArgumentParser()
united_parser.add_argument("--model_united", default="ogdlc", type=str, help="specific model: [ogdlc,masked_ftml,twp,genolc,adpolc,SDN, MSDNet, l2stop]")
# united_parser.add_argument("--em", default=0, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")
# united_parser.add_argument("--validation", default=25000, type=int, help="validate at this interval (default 25000)")
# united_parser.add_argument("--dataset", "-d", default="svhn", type=str, help="dataset name : [svhn, cifar10]")
# united_parser.add_argument("--root", "-r", default="data", type=str, help="dataset dir")
# united_parser.add_argument("--output", "-o", default="./exp_res", type=str, help="output dir")
united_args = united_parser.parse_args()

# #model:MSDNet
# import MSDNet.main as MSDNet_model
# from MSDNet.args import arg_parser as MSDNet_parser
# MSDNet_param = MSDNet_parser.parse_args()
# import MSDNet.models as models

# #model:SDN
# import SDN_official_unified.train_networks as SDN_model
# from SDN_official_unified.cmd_args_SDN import arg_parser as SDN_parser
# SDN_param = SDN_parser.parse_args()

# #model:l2stop
# import sdn_stop.train_networks as SDN_model_l2stop
# from sdn_stop.cmd_args_l2stop import cmd_opt as l2stop_parser
# l2stop_param = l2stop_parser.parse_args()
# from sdn_stop.train_stop_kl import policy_main
# from sdn_stop.models import MulticlassNetImage, MNIconfidence, Imiconfidence

class concise_log:
    def __init__(self, log_fname):
        self.fname = log_fname
        self.logf = open(log_fname, 'a')
        self.logf.write("\n***************************************\n")
        self.logf.close()

    def write(self, line):
        self.logf = open(self.fname, 'a')
        self.logf.write(line)
        self.logf.write("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        self.logf.close()

    def close(self):
        self.logf.close()

def create_path(dir):
    if dir is not None:
        if not os.path.isdir(dir):
            os.makedirs(dir)

# to log the output of the experiments to a file
class Logger(object):
    def __init__(self, log_file, mode='out'):
        if mode == 'out':
            self.terminal = sys.stdout
        else:
            self.terminal = sys.stderr
        self.log = open('{}.{}'.format(log_file, mode), "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()

def set_logger(log_file):
    sys.stdout = Logger(log_file, 'out')
    #sys.stderr = Logger(log_file, 'err')

class syn_settings:
    def __init__(self):
        self.save_dir       = None
        self.log_file       = None
        self.seed           = None
        # self.GPUs           = None
        # self.print_freq     = None
        self.dataset        = None
        self.dataset_root   = None
        # self.model_names    = None
        self.model_name     = None
        # self.backbone       = None
        self.model_settings = {}

    def summary(self):
        settings_str = ""
        for i, (key, value) in enumerate(self.__dict__.items()):
            # print(key)
            # print(value,type(value))
            # if value == '-1': continue
            param = str(key)
            param_value = str(value)
            settings_str += "| {} = {}\n".format(param.ljust(20), param_value)
            # print("| {} = {}".format(param.ljust(20), param_value))
        return settings_str

    # def _try(self, o):
    #     try:
    #         return o.__dict__
    #     except:
    #         return str(o)
    #
    # def to_JSON(self):
    #     return json.dumps(self, default=lambda o: self._try(o), sort_keys=True, indent=4)

class model_setting:
    def __init__(self):
        self.val_batch_size            = None
        self.K = None
        self.Kq     = None
        self.inner_steps         = None
        self.num_iterations        = None
        self.meta_batch     = None
        self.eta_1 = None
        self.eta_3  = None
        self.num_neighbors = None
        self.d_feature = None
        self.log_file = None
    def summary(self):
        settings_str = ""
        for i, (key, value) in enumerate(self.__dict__.items()):
            if value == -1: continue # to filter out the parameter for other models
            param = str(key)
            param_value = str(value)
            settings_str += "| {} = {}\n".format(param.ljust(20), param_value)
        return settings_str

def read_model_setting(model_name='masked_ftml'):
    if model_name == 'masked_ftml':
        model_params = config.masked_ftml
    if model_name == 'twp':
        model_params = config.twp
    if model_name == 'genolc':
        model_params = config.genolc
    if model_name == 'adpolc':
        model_params = config.adpolc
    if model_name == 'ogdlc':
        model_params = config.ogdlc
    # if model_name == 'SDN':
    #     model_params = config.SDN
    # if model_name == 'MSDNet':
    #     model_params = config.MSDNet
    # if model_name == 'l2stop':
    #     model_params = config.l2stop
    model_params = defaultdict(lambda :-1, model_params)
    model_set                   = model_setting()
    model_set.log_file = model_params["log_file"]
    model_set.val_batch_size          = model_params["val_batch_size"]
    model_set.K            = model_params["K"]
    model_set.Kq     = model_params["Kq"]
    model_set.inner_steps         = model_params["inner_steps"]
    model_set.num_iterations        = model_params["num_iterations"]
    model_set.meta_batch     = model_params["meta_batch"]
    model_set.eta_1 = model_params["eta_1"]
    model_set.eta_3  = model_params["eta_3"]
    model_set.eps = model_params["eps"]
    model_set.num_neighbors = model_params["num_neighbors"]
    model_set.d_feature = model_params["d_feature"]
    model_set.delta = model_params["delta"]

    return model_set

def read_settings(display=True):
    settings = syn_settings()
    params = config.General_Parameters

    settings.save_dir     = params["save_dir"]
    settings.log_file     = params["log_file"]
    settings.seed         = params["seed"]
    # settings.GPUs         = params["GPUs"]
    # settings.print_freq   = params["print_freq"]
    settings.dataset      = params["dataset"]
    settings.dataset_root = params["dataset_root"]
    settings.model_name   = united_args.model_united
    # settings.backbone     = params["backbone"]

    # for model_name in settings.model_names:
    model_setting = read_model_setting(settings.model_name)
    settings.model_settings[settings.model_name] = model_setting

    if display:
        print('General Parameters:\n', settings.summary())
        # for model_name in settings.model_names:
        print(f'Specific Model Parameters for {settings.model_name}:\n', settings.model_settings[settings.model_name].summary())
        # print("****", settings.model_settings[settings.model_name].log_file)
    return settings

def list_2_str(lis):
    lis_str = [str(x) for x in lis]
    return str(",".join(lis_str))
def masked_ftml_param_convert(settings, model_param):
    model_param.save          = settings.save_dir+ "/masked_ftml"
    model_param.log_file      = settings.log_file
    model_param.seed          = settings.seed
    model_param.data          = settings.dataset[0]
    model_param.data_root     = settings.dataset_root
    model_param.model_name = settings.model_name
    model_param.val_batch_size = settings.model_settings[settings.model_name].val_batch_size
    model_param.K = settings.model_settings[settings.model_name].K
    model_param.Kq = settings.model_settings[settings.model_name].Kq
    model_param.inner_steps = settings.model_settings[settings.model_name].inner_steps
    model_param.num_iterations = settings.model_settings[settings.model_name].num_iterations
    model_param.meta_batch = settings.model_settings[settings.model_name].meta_batch
    model_param.eta_1 = settings.model_settings[settings.model_name].eta_1
    model_param.eta_3 = settings.model_settings[settings.model_name].eta_3
    model_param.num_neighbors = settings.model_settings[settings.model_name].num_neighbors
    model_param.d_feature = settings.model_settings[settings.model_name].d_feature

    return model_param

def twp_param_convert(settings, model_param):
    model_param.save          = settings.save_dir+ "/twp"
    model_param.log_file      = settings.log_file
    model_param.seed          = settings.seed
    model_param.data          = settings.dataset[0]
    model_param.data_root     = settings.dataset_root
    model_param.model_name = settings.model_name
    # print(f'Specific Model Parameters for {settings.model_name}:\n',
    #       settings.model_settings[settings.model_name].summary())
    model_param.val_batch_size = settings.model_settings[settings.model_name].val_batch_size
    model_param.K = settings.model_settings[settings.model_name].K
    model_param.Kq = settings.model_settings[settings.model_name].Kq
    model_param.inner_steps = settings.model_settings[settings.model_name].inner_steps
    model_param.num_iterations = settings.model_settings[settings.model_name].num_iterations
    model_param.meta_batch = settings.model_settings[settings.model_name].meta_batch
    model_param.eta_1 = settings.model_settings[settings.model_name].eta_1
    model_param.eps = settings.model_settings[settings.model_name].eps
    model_param.num_neighbors = settings.model_settings[settings.model_name].num_neighbors
    model_param.d_feature = settings.model_settings[settings.model_name].d_feature

    return model_param
def genolc_param_convert(settings, model_param):
    model_param.save          = settings.save_dir+ "/genolc"
    model_param.log_file      = settings.log_file
    model_param.seed          = settings.seed
    model_param.data          = settings.dataset[0]
    model_param.data_root     = settings.dataset_root
    model_param.model_name = settings.model_name
    # print(f'Specific Model Parameters for {settings.model_name}:\n',
    #       settings.model_settings[settings.model_name].summary())
    model_param.val_batch_size = settings.model_settings[settings.model_name].val_batch_size
    model_param.K = settings.model_settings[settings.model_name].K
    # model_param.inner_steps = settings.model_settings[settings.model_name].inner_steps
    model_param.num_iterations = settings.model_settings[settings.model_name].num_iterations
    model_param.delta = settings.model_settings[settings.model_name].delta
    model_param.eta_1 = settings.model_settings[settings.model_name].eta_1
    model_param.eps = settings.model_settings[settings.model_name].eps
    model_param.num_neighbors = settings.model_settings[settings.model_name].num_neighbors
    model_param.d_feature = settings.model_settings[settings.model_name].d_feature


    return model_param
def adpolc_param_convert(settings, model_param):
    model_param.save          = settings.save_dir+ "/adpolc"
    model_param.log_file      = settings.log_file
    model_param.seed          = settings.seed
    model_param.data          = settings.dataset[0]
    model_param.data_root     = settings.dataset_root
    model_param.model_name = settings.model_name
    model_param.val_batch_size = settings.model_settings[settings.model_name].val_batch_size
    model_param.K = settings.model_settings[settings.model_name].K
    model_param.num_iterations = settings.model_settings[settings.model_name].num_iterations
    model_param.eps = settings.model_settings[settings.model_name].eps
    model_param.num_neighbors = settings.model_settings[settings.model_name].num_neighbors
    model_param.d_feature = settings.model_settings[settings.model_name].d_feature

def ogdlc_param_convert(settings, model_param):
    model_param.save = settings.save_dir + "/ogdlc"
    model_param.log_file = settings.log_file
    model_param.seed = settings.seed
    model_param.data = settings.dataset[0]
    model_param.data_root = settings.dataset_root
    model_param.model_name = settings.model_name
    model_param.val_batch_size = settings.model_settings[settings.model_name].val_batch_size
    model_param.K = settings.model_settings[settings.model_name].K
    model_param.eta_1 = settings.model_settings[settings.model_name].eta_1
    model_param.eps = settings.model_settings[settings.model_name].eps
    model_param.delta = settings.model_settings[settings.model_name].delta
    model_param.num_neighbors = settings.model_settings[settings.model_name].num_neighbors
    model_param.d_feature = settings.model_settings[settings.model_name].d_feature
    return model_param
def MSDNet_param_convert(settings, model_param):
    model_param.save          = settings.save_dir+ "/MSDNet"
    model_param.log_file      = settings.log_file
    model_param.seed          = settings.seed
    model_param.gpu           = list_2_str(settings.GPUs[1])
    model_param.print_freq    = settings.print_freq
    model_param.data          = settings.dataset[0]
    model_param.data_root     = settings.dataset_root
    model_param.epochs        = settings.model_settings['MSDNet'].epochs
    model_param.lr            = settings.model_settings['MSDNet'].learning_rate
    model_param.optimizer     = settings.model_settings['MSDNet'].optimizer

    if model_param.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = model_param.gpu

    model_param.grFactor = list(map(int, model_param.grFactor.split('-')))
    model_param.bnFactor = list(map(int, model_param.bnFactor.split('-')))
    model_param.nScales = len(model_param.grFactor)

    if model_param.use_valid:
        model_param.splits = ['train', 'val', 'test']
    else:
        model_param.splits = ['train', 'val']

    if model_param.data == 'cifar10' or model_param.data == 'mnist':
        model_param.num_classes = 10
    elif model_param.data == 'cifar100':
        model_param.num_classes = 100
    else:
        model_param.num_classes = 1000
    return model_param

def SDN_param_convert(settings, model_param):
    model_param.model_united  = united_args.model_united
    model_param.save          = settings.save_dir + "/SDN"
    model_param.log_file      = settings.log_file
    model_param.seed          = settings.seed
    model_param.gpu_id        = settings.GPUs[0][0]
    model_param.print_freq    = settings.print_freq
    model_param.data          = settings.dataset[0]
    model_param.data_root     = settings.dataset_root
    model_param.epochs        = settings.model_settings['SDN'].epochs
    model_param.learning_rate = settings.model_settings['SDN'].learning_rate
    model_param.optimizer     = settings.model_settings['SDN'].optimizer

    model_param.add_ic_vgg        = settings.model_settings['SDN'].add_ic_vgg
    model_param.add_ic_resnet     = settings.model_settings['SDN'].add_ic_resnet
    model_param.add_ic_wideresnet = settings.model_settings['SDN'].add_ic_wideresnet
    model_param.add_ic_mobilenet  = settings.model_settings['SDN'].add_ic_mobilenet

    return model_param

def l2stop_param_convert(settings, model_param):
    model_param.save_dir          = settings.save_dir + "/l2stop"
    model_param.log_file          = settings.log_file
    model_param.seed              = settings.seed
    model_param.gpu_id            = settings.GPUs[2][0]
    model_param.print_freq        = settings.print_freq
    model_param.data              = settings.dataset[0]
    model_param.data_root         = settings.dataset_root
    model_param.epochs            = settings.model_settings['l2stop'].epochs
    model_param.epochs_policy_net = settings.model_settings['l2stop'].epochs_policy_net
    model_param.learning_rate     = settings.model_settings['l2stop'].learning_rate
    model_param.optimizer         = settings.model_settings['l2stop'].optimizer

    model_param.add_ic_vgg        = settings.model_settings['l2stop'].add_ic_vgg
    model_param.add_ic_resnet     = settings.model_settings['l2stop'].add_ic_resnet
    model_param.add_ic_wideresnet = settings.model_settings['l2stop'].add_ic_wideresnet
    model_param.add_ic_mobilenet  = settings.model_settings['l2stop'].add_ic_mobilenet

    return model_param

def load_SDN_params(models_path, model_name, epoch=0):
    params_path = models_path + '/' + model_name
    if epoch == 0:
        params_path = params_path + '/parameters_untrained'
    else:
        params_path = params_path + '/parameters_last'

    with open(params_path, 'rb') as f:
        model_params = pickle.load(f)
    return model_params

def load_SDN_model(models_path, model_name, epoch=0):
    from SDN_official_unified.architectures.SDNs.VGG_SDN import VGG_SDN
    model_params = load_SDN_params(models_path, model_name, epoch)
    model = VGG_SDN(model_params)
    network_path = models_path + '/' + model_name

    if epoch == 0: # untrained model
        load_path = network_path + '/untrained'
    elif epoch == -1: # last model
        load_path = network_path + '/last'
    else:
        load_path = network_path + '/' + str(epoch)

    model.load_state_dict(torch.load(load_path), strict=False)
    return model, model_params

def MSDNet_get_detailed_results(args, model, split, device='cpu'):
    from MSDNet.dataloader import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(args)
    if split == 'train':
        loader = train_loader
    if split == 'valid':
        loader = val_loader
    if split == 'test':
        loader = test_loader

    model.to(device)
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    layer_predictions = {}
    layer_confidence = {}

    is_correct_all_list = []
    loss_layer_all_list = []

    criterion = nn.CrossEntropyLoss(reduction='none')

    outputs = list(range(args.nBlocks)) #i.e.: [0,1,2,3] TODO: is this the num of internal classifiers?

    for output_id in outputs:
        layer_correct[output_id] = set()  #{0: set(), 1: set(), 2: set(), 3: set(), 4: set(), 5: set(), 6: set()}
        layer_wrong[output_id] = set()
        layer_predictions[output_id] = {} #{0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}}
        layer_confidence[output_id] = {}

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x) #list:4, i.e.: output[0]: Tensor 64*10, similar for 1, 2, 3
            output_sm = [nn.functional.softmax(out, dim=1) for out in output]

            loss_layer_batch_list = []
            is_correct_batch_list = []

            for output_id in outputs:
                cur_output = output[output_id] #each layer
                cur_loss = criterion(cur_output, b_y)
                loss_layer_batch_list.append(cur_loss) # i.e.: list of torch.Size([128])

                cur_confidences = output_sm[output_id].max(1, keepdim=True)[0]
                pred = cur_output.max(1, keepdim=True)[1]

                is_correct = pred.eq(b_y.view_as(pred)) #torch.Size([128, 1])
                is_correct_batch_list.append(is_correct)

                for test_id in range(len(b_x)):
                    #each instance
                    cur_instance_id = test_id + cur_batch_id*loader.batch_size
                    correct = is_correct[test_id]
                    layer_predictions[output_id][cur_instance_id] = pred[test_id].cpu().numpy()
                    layer_confidence[output_id][cur_instance_id] = cur_confidences[test_id].cpu().numpy()
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)

            loss_layer_batch = torch.stack(loss_layer_batch_list, dim=1)  # torch.Size([128, 7])
            loss_layer_all_list.append(loss_layer_batch)

            is_correct_batch = torch.cat(is_correct_batch_list, dim=1)  # torch.Size([128, 7])
            is_correct_all_list.append(is_correct_batch)

        loss_layer_all = torch.cat(loss_layer_all_list) #10000 * 7
        is_correct_all = torch.cat(is_correct_all_list) #10000 * 7, is correct or not
    return layer_correct, layer_wrong, layer_predictions, layer_confidence, is_correct_all, loss_layer_all

def sdn_get_detailed_results(args, model, split, device='cpu'):
    import SDN_official_unified.aux_funcs as afun
    dataloader = afun.get_dataset(args.data)
    if split == 'train':
        loader = dataloader.aug_train_loader
    if split == 'valid':
        loader = dataloader.valid_loader
    if split == 'test':
        loader = dataloader.test_loader

    model.to(device)
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    layer_predictions = {}
    layer_confidence = {}

    is_correct_all_list = []
    loss_layer_all_list = []

    criterion = nn.CrossEntropyLoss(reduction='none')

    outputs = list(range(model.num_output)) #i.e.: [0,1,2,3,4,5,6]

    for output_id in outputs:
        layer_correct[output_id] = set()  #{0: set(), 1: set(), 2: set(), 3: set(), 4: set(), 5: set(), 6: set()}
        layer_wrong[output_id] = set()
        layer_predictions[output_id] = {} #{0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}}
        layer_confidence[output_id] = {}

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output_sm = [nn.functional.softmax(out, dim=1) for out in output]

            loss_layer_batch_list = []
            is_correct_batch_list = []

            for output_id in outputs:
                cur_output = output[output_id] #each layer
                cur_loss = criterion(cur_output, b_y)
                loss_layer_batch_list.append(cur_loss) # i.e.: list of torch.Size([128])

                cur_confidences = output_sm[output_id].max(1, keepdim=True)[0]
                pred = cur_output.max(1, keepdim=True)[1]

                is_correct = pred.eq(b_y.view_as(pred)) #torch.Size([128, 1])
                is_correct_batch_list.append(is_correct)

                for test_id in range(len(b_x)):
                    #each instance
                    cur_instance_id = test_id + cur_batch_id*loader.batch_size
                    correct = is_correct[test_id]
                    layer_predictions[output_id][cur_instance_id] = pred[test_id].cpu().numpy()
                    layer_confidence[output_id][cur_instance_id] = cur_confidences[test_id].cpu().numpy()
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)

            loss_layer_batch = torch.stack(loss_layer_batch_list, dim=1)  # torch.Size([128, 7])
            loss_layer_all_list.append(loss_layer_batch)

            is_correct_batch = torch.cat(is_correct_batch_list, dim=1)  # torch.Size([128, 7])
            is_correct_all_list.append(is_correct_batch)

        loss_layer_all = torch.cat(loss_layer_all_list) #10000 * 7
        is_correct_all = torch.cat(is_correct_all_list) #10000 * 7, is correct or not
    return layer_correct, layer_wrong, layer_predictions, layer_confidence, is_correct_all, loss_layer_all

def precision_recall_fscore(adjust_param, model, split, device):
    layer_corr, layer_wrong, layer_preds, layer_confidence, is_correct_all, loss_layer_all = sdn_get_detailed_results(
        adjust_param, model, split, device)

    confidence_layer_all = torch.zeros_like(is_correct_all, dtype=torch.double)
    for layer_id, layer_conf in layer_confidence.items():
        for sample_id, sample_conf in layer_conf.items():
            confidence_layer_all[sample_id][layer_id] = sample_conf[0].astype(np.float64)

    highest_conf_value, highest_conf_index = torch.max(confidence_layer_all, 1)   # TODO: upgrade pytorch 1.6 to a higher version
                                                                                  # because the current version has some problem:
                                                                                  # max and argmax are not consistent

    stop_layer_pred = highest_conf_index  # select the highest confidence layer as the predicted stopping layer
    stop_layer_true = loss_layer_all.argmin(1)  # the layer with the lowest loss as the true stopping layer

    target_names = ['layer_0', 'layer_1', 'layer_2', 'layer_3', 'layer_4', 'layer_5', 'layer_6']
    classifi_report = classification_report(stop_layer_true.cpu().numpy(), stop_layer_pred.cpu().numpy(),
                                            target_names=target_names)

    # accuracy of selecting the highest confidence score layer
    highest_conf_layer_all = torch.zeros_like(is_correct_all)
    highest_conf_layer_all[torch.arange(is_correct_all.shape[0]), highest_conf_index] = 1
    is_correct_highest_conf = highest_conf_layer_all * is_correct_all
    accuracy_overall = torch.true_divide(torch.sum(torch.sum(is_correct_highest_conf)).item(),is_correct_all.shape[0]) # overall accuracy

    accuracy_overall = "%.4f" % accuracy_overall.item()
    return classifi_report, accuracy_overall

def load_MSDNet_model(args, saved_path):
    #1. define the model
    model = getattr(models, args.arch)(args) #the output is clear, but how does the getattr function work?
    model = torch.nn.DataParallel(model).cuda()

    #2. load the saved state
    state = torch.load(saved_path)

    #3. load the state_dict
    model.load_state_dict(state['state_dict'])
    return model

def precision_recall_fscore_MSDNet(adjust_param, model, split, device):
    layer_corr, layer_wrong, layer_preds, layer_confidence, is_correct_all, loss_layer_all = MSDNet_get_detailed_results(
        adjust_param, model, split, device)

    confidence_layer_all = torch.zeros_like(is_correct_all, dtype=torch.double)
    for layer_id, layer_conf in layer_confidence.items():
        for sample_id, sample_conf in layer_conf.items():
            confidence_layer_all[sample_id][layer_id] = sample_conf[0].astype(np.float64)

    highest_conf_value, highest_conf_index = torch.max(confidence_layer_all, 1)   # TODO: upgrade pytorch 1.6 to a higher version
                                                                                  # TODO: because the current version has some problem:
                                                                                  # TODO: max and argmax are not consistent

    stop_layer_pred = highest_conf_index  # select the highest confidence layer as the predicted stopping layer
    stop_layer_true = loss_layer_all.argmin(1)  # the layer with the lowest loss as the true stopping layer

    target_names = ['block_0', 'block_1', 'block_2', 'block_3']
    classifi_report = classification_report(stop_layer_true.cpu().numpy(), stop_layer_pred.cpu().numpy(),
                                            target_names=target_names)

    # accuracy of selecting the highest confidence score layer
    highest_conf_layer_all = torch.zeros_like(is_correct_all)
    highest_conf_layer_all[torch.arange(is_correct_all.shape[0]), highest_conf_index] = 1
    is_correct_highest_conf = highest_conf_layer_all * is_correct_all
    accuracy_overall = torch.true_divide(torch.sum(torch.sum(is_correct_highest_conf)).item(),
                                         is_correct_all.shape[0])  # overall accuracy

    accuracy_overall = "%.4f" % accuracy_overall.item()
    return classifi_report, accuracy_overall

# if __name__ == "__main__":

# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
print("Model: ", united_args.model_united)


settings = read_settings(display=True)
settings.model_name = united_args.model_united
create_path(settings.save_dir)
set_logger(os.path.join(settings.save_dir, "settings.log_file"))
print(f'Specific Model Parameters for {settings.model_name}:\n', settings.model_settings[settings.model_name].summary())
concise_log_file = settings.model_settings[settings.model_name].log_file
print(settings.save_dir,"  ",concise_log_file)
con_log = concise_log(os.path.join(settings.save_dir, concise_log_file))
con_log.write(f"************************* {time.strftime('%Y-%m-%d_%H-%M-%S')} *****************************\n")

con_log.write("General Parameters:")
con_log.write(settings.summary())

for model_name, _ in settings.model_settings.items():
    con_log.write(f"Specific Model Parameters for {model_name}:")
    con_log.write(settings.model_settings[model_name].summary())

# for model in settings.model_names:
if settings.model_name == 'masked_ftml':
    from cls_masked_ftml.m_ftml import  *
    from cls_masked_ftml.m_ftml_main import *
    from cls_masked_ftml.m_ftml_main import arg_parser as Masked_ftml_parser
    masked_ftml_para = Masked_ftml_parser
    adjust_param = masked_ftml_param_convert(settings,masked_ftml_para)

    print("Adjusted model params:\n", vars(adjust_param))
    print("\n=================== Model: masked_ftml ===================\n")
    res ,res_check,aucs= m_ftml_run(masked_ftml_para)
    saved_path = f'{adjust_param.save}/save_models/model_best.pth.tar'
    con_log.write('auc score for final task: '+str(aucs[-1]))
    with open(os.path.join(adjust_param.save, "results.pkl"), "wb") as f_write:
        pickle.dump(res, f_write, protocol=pickle.HIGHEST_PROTOCOL)

if settings.model_name == 'twp':
    from cls_twp.twp import *
    from cls_twp.twp_main import *
    from cls_twp.twp_main import arg_parser as twp_parser

    twp_para = twp_parser

    adjust_param = twp_param_convert(settings, twp_para)
    print("Adjusted model params:\n", vars(adjust_param))
    print("\n=================== Model: twp ===================\n")
    res, res_check,aucs = twp_run(twp_para)
    saved_path = f'{adjust_param.save}/save_models/model_best.pth.tar'
    con_log.write('auc score for final task: '+str(aucs[-1]))
    with open(os.path.join(adjust_param.save, "results.pkl"), "wb") as f_write:
        pickle.dump(res, f_write, protocol=pickle.HIGHEST_PROTOCOL)

if settings.model_name == 'genolc':
    from cls_genolc import *
    from cls_genolc.genolc_main import *
    from cls_genolc.genolc_main import arg_parser as genolc_parser

    gen_para = genolc_parser
    adjust_param = genolc_param_convert(settings, gen_para)
    print("Adjusted model params:\n", vars(adjust_param))
    print("\n=================== Model: genolc ===================\n")
    res, res_check,aucs = genolc_run(genolc_parser)
    saved_path = f'{adjust_param.save}/save_models/model_best.pth.tar'
    con_log.write('auc score for final task: '+str(aucs[-1]))
    with open(os.path.join(adjust_param.save, "results.pkl"), "wb") as f_write:
        pickle.dump(res, f_write, protocol=pickle.HIGHEST_PROTOCOL)
if settings.model_name == 'adpolc':
    from cls_adpolc import *
    from cls_adpolc.adpolc_main import *
    from cls_adpolc.adpolc_main import arg_parser as adpolc_parser

    adp_para = adpolc_parser
    adjust_param = adpolc_param_convert(settings, adp_para)
    print("Adjusted model params:\n", vars(adjust_param))
    print("\n=================== Model: adpolc ===================\n")
    res, res_check,aucs = adpolc_run(adp_para)
    # print("aucs",aucs)
    saved_path = f'{adjust_param.save}/save_models/model_best.pth.tar'

    con_log.write('auc score for final task: '+str(aucs[-1]))
    with open(os.path.join(adjust_param.save, "results.pkl"), "wb") as f_write:
        pickle.dump(res, f_write, protocol=pickle.HIGHEST_PROTOCOL)
if settings.model_name == 'ogdlc':
    from cls_ogdlc import *
    from cls_ogdlc.ogdlc_main import *
    from cls_ogdlc.ogdlc_main import arg_parser as ogdlc_parser

    ogd_para = ogdlc_parser
    adjust_param = ogdlc_param_convert(settings, ogd_para)
    print("Adjusted model params:\n", vars(adjust_param))
    print("\n=================== Model: adpolc ===================\n")
    res,aucs = ogdlc_run(ogd_para)
    # print("aucs",aucs)
    saved_path = f'{adjust_param.save}/save_models/model_best.pth.tar'

    con_log.write('auc score for final task: '+str(aucs[-1]))
    with open(os.path.join(adjust_param.save, "results.pkl"), "wb") as f_write:
        pickle.dump(res, f_write, protocol=pickle.HIGHEST_PROTOCOL)

# if united_args.model_united == "MSDNet":
    #import model related modules for model:MSDNet
    import MSDNet.main as MSDNet_model
    from MSDNet.args import arg_parser as MSDNet_parser
    import MSDNet.models as models
    MSDNet_param = MSDNet_parser.parse_args()

    print("\n=================== Model: MSDNet ===================\n")
    adjust_param = MSDNet_param_convert(settings, MSDNet_param)
    print("epochs:", adjust_param.epochs)
    print("Adjusted model params:\n", vars(adjust_param))
    start = time.time()
    MSDNet_model.main(adjust_param)
    con_log.write(f"Training time used: {time.time() - start} seconds")

    # saved_path = 'saved_results_0629/save_models/model_best.pth.tar'
    saved_path = f'{adjust_param.save}/save_models/model_best.pth.tar'
    model = load_MSDNet_model(adjust_param, saved_path)  # the most recent model
    device = 'cuda'
    classifi_report_tr, accuracy_overall_tr     = precision_recall_fscore_MSDNet(adjust_param, model, 'train',
                                                                                 device)
    print("\nmetrics of policy network in training set:\n", classifi_report_tr)

    classifi_report_val, accuracy_overall_val   = precision_recall_fscore_MSDNet(adjust_param, model, 'valid',
                                                                                 device)
    print("\nmetrics of policy network in validation set:\n", classifi_report_val)

    classifi_report_test, accuracy_overall_test = precision_recall_fscore_MSDNet(adjust_param, model, 'test',
                                                                                 device)
    print("\nmetrics of policy network in test set:\n", classifi_report_test)

    con_log.write("Overall accuracy in training/Validation/Test set:")
    con_log.write(", ".join([str(accuracy_overall_tr), str(accuracy_overall_val), str(accuracy_overall_test)]))

    con_log.write("precision, recall, fscore of policy network in training set:")
    con_log.write(classifi_report_tr)

    con_log.write("precision, recall, fscore of policy network in validation set:")
    con_log.write(classifi_report_val)

    con_log.write("precision, recall, fscore of policy network in test set:")
    con_log.write(classifi_report_test)

    con_log.write(f"Total time used: {time.time() - start} seconds")

    classifi_rpt = (classifi_report_tr, classifi_report_val, classifi_report_test)
    overall_acc  = (accuracy_overall_tr, accuracy_overall_val, accuracy_overall_test)
    with open(os.path.join(adjust_param.save, "results.pkl"), "wb") as f_write:
        pickle.dump((classifi_rpt, overall_acc), f_write, protocol=pickle.HIGHEST_PROTOCOL)

if settings.model_name == 'MSDNet':
# if united_args.model_united == "MSDNet":
    #import model related modules for model:MSDNet
    import MSDNet.main as MSDNet_model
    from MSDNet.args import arg_parser as MSDNet_parser
    import MSDNet.models as models
    MSDNet_param = MSDNet_parser.parse_args()

    print("\n=================== Model: MSDNet ===================\n")
    adjust_param = MSDNet_param_convert(settings, MSDNet_param)
    print("epochs:", adjust_param.epochs)
    print("Adjusted model params:\n", vars(adjust_param))
    start = time.time()
    MSDNet_model.main(adjust_param)
    con_log.write(f"Training time used: {time.time() - start} seconds")

    # saved_path = 'saved_results_0629/save_models/model_best.pth.tar'
    saved_path = f'{adjust_param.save}/save_models/model_best.pth.tar'
    model = load_MSDNet_model(adjust_param, saved_path)  # the most recent model
    device = 'cuda'
    classifi_report_tr, accuracy_overall_tr     = precision_recall_fscore_MSDNet(adjust_param, model, 'train',
                                                                                 device)
    print("\nmetrics of policy network in training set:\n", classifi_report_tr)

    classifi_report_val, accuracy_overall_val   = precision_recall_fscore_MSDNet(adjust_param, model, 'valid',
                                                                                 device)
    print("\nmetrics of policy network in validation set:\n", classifi_report_val)

    classifi_report_test, accuracy_overall_test = precision_recall_fscore_MSDNet(adjust_param, model, 'test',
                                                                                 device)
    print("\nmetrics of policy network in test set:\n", classifi_report_test)

    con_log.write("Overall accuracy in training/Validation/Test set:")
    con_log.write(", ".join([str(accuracy_overall_tr), str(accuracy_overall_val), str(accuracy_overall_test)]))

    con_log.write("precision, recall, fscore of policy network in training set:")
    con_log.write(classifi_report_tr)

    con_log.write("precision, recall, fscore of policy network in validation set:")
    con_log.write(classifi_report_val)

    con_log.write("precision, recall, fscore of policy network in test set:")
    con_log.write(classifi_report_test)

    con_log.write(f"Total time used: {time.time() - start} seconds")

    classifi_rpt = (classifi_report_tr, classifi_report_val, classifi_report_test)
    overall_acc  = (accuracy_overall_tr, accuracy_overall_val, accuracy_overall_test)
    with open(os.path.join(adjust_param.save, "results.pkl"), "wb") as f_write:
        pickle.dump((classifi_rpt, overall_acc), f_write, protocol=pickle.HIGHEST_PROTOCOL)

if settings.model_name == 'SDN':
# if united_args.model_united == "SDN":
#     settings.model_name = united_args.model_united
    # model:SDN
    import SDN_official_unified.train_networks as SDN_model
    from SDN_official_unified.cmd_args_SDN import arg_parser as SDN_parser
    SDN_param = SDN_parser.parse_args()

    model_flag = "\n=================== Model: SDN ===================\n"
    print(model_flag)
    con_log.write(model_flag)
    adjust_param = SDN_param_convert(settings, SDN_param)
    print("epochs:", adjust_param.epochs)
    print("Adjusted model params:\n", vars(adjust_param))
    start = time.time()
    SDN_model.train_models(adjust_param)
    con_log.write(f"Training time used: {time.time() - start} seconds")

    # load the most recent model / the best model
    model_name = 'mnist_vgg16bn_sdn_sdn_training'
    model, params = load_SDN_model(adjust_param.save, model_name, epoch=-1)  # the most recent model
    device = 'cuda'
    classifi_report_tr, accuracy_overall_tr   = precision_recall_fscore(adjust_param, model, 'train', device)
    print("\nmetrics of policy network in training set:\n", classifi_report_tr)

    classifi_report_val, accuracy_overall_val = precision_recall_fscore(adjust_param, model, 'valid', device)
    print("\nmetrics of policy network in validation set:\n", classifi_report_val)

    classifi_report_test, accuracy_overall_test = precision_recall_fscore(adjust_param, model, 'test', device)
    print("\nmetrics of policy network in test set:\n", classifi_report_test)

    con_log.write("Overall accuracy in training/Validation/Test set:")
    con_log.write(", ".join([str(accuracy_overall_tr), str(accuracy_overall_val), str(accuracy_overall_test)]))

    con_log.write("precision, recall, fscore of policy network in training set:")
    con_log.write(classifi_report_tr)

    con_log.write("precision, recall, fscore of policy network in validation set:")
    con_log.write(classifi_report_val)

    con_log.write("precision, recall, fscore of policy network in test set:")
    con_log.write(classifi_report_test)

    con_log.write(f"Total time used: {time.time() - start} seconds")

    classifi_rpt = (classifi_report_tr, classifi_report_val, classifi_report_test)
    overall_acc = (accuracy_overall_tr, accuracy_overall_val, accuracy_overall_test)
    with open(os.path.join(adjust_param.save, "results.pkl"), "wb") as f_write:
        pickle.dump((classifi_rpt, overall_acc), f_write, protocol=pickle.HIGHEST_PROTOCOL)

if settings.model_name == 'l2stop':
# if united_args.model_united == "l2stop":
    #TODO: load model needs to optimize
    # model:l2stop
    import sdn_stop.train_networks as SDN_model_l2stop
    from sdn_stop.cmd_args_l2stop import cmd_opt as l2stop_parser
    from sdn_stop.train_stop_kl import policy_main
    l2stop_param = l2stop_parser.parse_args()

    model_flag = "\n=================== Model: learn to stop ===================\n"
    print(model_flag)
    con_log.write(model_flag)
    adjust_param = l2stop_param_convert(settings, l2stop_param)
    print("epochs:", adjust_param.epochs)
    print("Adjusted model params:\n", vars(adjust_param))

    #step 1: SDN model
    start = time.time()
    SDN_model_l2stop.train_models(adjust_param)
    con_log.write(f"Training time of backbone model used: {time.time() - start} seconds")

    #step 2: train policy network
    start_policy = time.time()
    classifi_report, accuracy_overall, training_time = policy_main(adjust_param)
    con_log.write(f"Training time of policy network used: {training_time} seconds")

    con_log.write("Overall accuracy in training/Validation/Test set:")
    con_log.write(", ".join([str(accuracy_overall[0]), str(accuracy_overall[1]), str(accuracy_overall[2])]))

    con_log.write("precision, recall, fscore of policy network in training set:")
    con_log.write(classifi_report[0])

    con_log.write("precision, recall, fscore of policy network in validation set:")
    con_log.write(classifi_report[1])

    con_log.write("precision, recall, fscore of policy network in test set:")
    con_log.write(classifi_report[2])

    con_log.write(f"Total time used: {time.time() - start} seconds")

    with open(os.path.join(adjust_param.save_dir, "results.pkl"), "wb") as f_write:
        pickle.dump((classifi_report, accuracy_overall), f_write, protocol=pickle.HIGHEST_PROTOCOL)


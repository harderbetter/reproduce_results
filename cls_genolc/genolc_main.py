import time
from os import listdir
from os.path import isfile, join

from .genolc import *

class arg_parser:
    def __init__(self):
        self.val_batch_size            = None
        self.K = None
        # self.Kq     = None
        self.inner_steps         = None
        self.num_iterations        = None
        self.delta     = None
        self.eta_1 = None
        self.eps  = None
        self.num_neighbors = None
        self.d_feature = None
        self.data_root = None
        self.data = None
        self.save = None
    def summary(self):
        settings_str = ""
        for i, (key, value) in enumerate(self.__dict__.items()):
            param = str(key)
            param_value = str(value)
            settings_str += "| {} = {}\n".format(param.ljust(20), param_value)
        return settings_str
def genolc_run(arg_parser):
    print("enter_genolc")
    lamb = random.choice([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
    val_batch_size = arg_parser.val_batch_size  # data points for validation

    eps = arg_parser.eps
    eta_1 = arg_parser.eta_1
    delta = arg_parser.delta
    num_neighbors = arg_parser.num_neighbors
    K = arg_parser.K

    # cls_syn_data:2; adult:16; communities_and_crime:100; bank:16; census_income:36
    # cls_syn_data:2; adult:16; communities_and_crime:100; bank:16; census_income:36
    data_path = arg_parser.data_root
    # dataset = r'bank'
    dataset = arg_parser.data
    datasetALL = ["ny_stop_and_frisk", 'communities_and_crime']
    d_feature = arg_parser.d_feature[datasetALL.index(dataset)]  # feature size of the data set
    save = arg_parser.save
    tasks = [x[0] for x in os.walk(data_path + '/' + dataset)][1:]

    start = time.time()
    print(lamb)
    num_iterations = arg_parser.num_iterations
    res,res_check,aucs =genolc(d_feature, lamb, tasks, data_path, dataset, save,
           K, val_batch_size, num_neighbors,
           eta_1, delta, eps,num_iterations)
    cost_time_in_second = time.time() - start
    cost_time = time.strftime("%H:%M:%S", time.gmtime(cost_time_in_second))
    print(cost_time)
    return res,aucs
if __name__ == "__main__":
    # [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    lamb = random.choice([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
    # lamb=1000
    K = 100
    val_batch_size = 0.9  # data points for validation

    eps = 0.5
    eta_1 = 20
    delta = 50
    num_neighbors = 3

    # cls_syn_data:2; adult:16; communities_and_crime:100; bank:16; census_income:36
    # cls_syn_data:2; adult:16; communities_and_crime:100; bank:16; census_income:36
    d_feature = 51  # feature size of the data set
    data_path = r'/home/feng/FFML/data'
    # dataset = r'bank'
    dataset = r'ny_stop_and_frisk'
    save = r'/home/feng/FFML/save'
    tasks = [x[0] for x in os.walk(data_path + '/' + dataset)][1:]

    start = time.time()
    print(lamb)
    genolc(d_feature, lamb, tasks, data_path, dataset, save,
         K, val_batch_size, num_neighbors,
         eta_1, delta, eps)
    cost_time_in_second = time.time() - start
    cost_time = time.strftime("%H:%M:%S", time.gmtime(cost_time_in_second))
    print(cost_time)

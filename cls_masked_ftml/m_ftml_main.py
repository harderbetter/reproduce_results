import time
from os import listdir
from os.path import isfile, join
import argparse
from .m_ftml import *


class arg_parser:
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
def m_ftml_run(arg_parser):
    # parser = argparse.ArgumentParser(description='Process some integers.')
    print("asdada")
    val_batch_size = arg_parser.val_batch_size  # data points for validation
    K = arg_parser.K  # few shots in support
    Kq = arg_parser.Kq  # shots for query
    inner_steps = arg_parser.inner_steps  # gradient steps in the inner loop
    num_iterations = arg_parser.num_iterations  # outer iteration
    meta_batch = arg_parser.meta_batch

    eta_1 = arg_parser.eta_1  # step size of inner primal update
    eta_3 = arg_parser.eta_3 # step size of outer primal update
    num_neighbors = arg_parser.num_neighbors

    # cls_syn_data:2; adult:16; communities_and_crime:100; bank:16; census_income:36
    d_feature = arg_parser.d_feature  # feature size of the data set
    data_path = arg_parser.data_root
    # dataset = r'bank'
    dataset = arg_parser.data
    save = arg_parser.save
    print("path",data_path + '/' + dataset)
    tasks = [x[0] for x in os.walk(data_path + '/' + dataset)][1:]
    print("tasks",tasks)

    start = time.time()
    res =mftml(d_feature, tasks, data_path, dataset, save,
         K, Kq, val_batch_size, num_neighbors,
         num_iterations, inner_steps, meta_batch,
         eta_1, eta_3)
    cost_time_in_second = time.time() - start
    cost_time = time.strftime("%H:%M:%S", time.gmtime(cost_time_in_second))
    print("cost_time",cost_time)
    print("res",res)
    res.append(["total cost_time :"+str(cost_time)])
    return res

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Process some integers.')
    val_batch_size = 0.9  # data points for validation
    K = 100  # few shots in support
    Kq = 2 * K  # shots for query
    inner_steps = 1  # gradient steps in the inner loop
    num_iterations = 20  # outer iteration
    meta_batch = 32

    eta_1 = 0.00001  # step size of inner primal update
    eta_3 = 0.00001  # step size of outer primal update
    num_neighbors = 3

    # cls_syn_data:2; adult:16; communities_and_crime:100; bank:16; census_income:36
    d_feature = 51  # feature size of the data set
    data_path = r'/home/feng/FFML/data'
    # dataset = r'bank'
    dataset = r'ny_stop_and_frisk'
    save = r'/home/feng/FFML/save'
    tasks = [x[0] for x in os.walk(data_path + '/' + dataset)][1:]

    start = time.time()
    mftml(d_feature, tasks, data_path, dataset, save,
         K, Kq, val_batch_size, num_neighbors,
         num_iterations, inner_steps, meta_batch,
         eta_1, eta_3)
    cost_time_in_second = time.time() - start
    cost_time = time.strftime("%H:%M:%S", time.gmtime(cost_time_in_second))
    print(cost_time)

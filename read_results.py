import pickle
import os
import time

class concise_log:
    def __init__(self, log_fname):
        self.fname = log_fname
        self.logf = open(log_fname, 'a')
        self.logf.write("\n***************************************\n")
        self.logf.close()

    def write(self, line):
        self.logf = open(self.fname, 'a')
        self.logf.write(line)
        self.logf.write("\n")
        self.logf.close()

    def close(self):
        self.logf.close()

path_0 = "saved_results_test_0712_3"
con_log = concise_log(os.path.join(path_0, "comparison.log"))
con_log.write(f"************************* comparisons among different models *****************************\n")
con_log.write(f"************************* {time.strftime('%Y-%m-%d_%H-%M-%S')} *****************************\n")
# models = ["SDN", "l2stop", "MSDNet"]
models = ["masked_ftml"]
for model in models:
    path = os.path.join(path_0, model)
    with open(os.path.join(path, "results.pkl"), "rb") as f:
        classifi_rpt, overall_accus = pickle.load(f)
        con_log.write(f"********Model {model}: Overall Training/Validation/Test Accuracy:\n")
        con_log.write(", ".join([str(overall_accus[0]), str(overall_accus[1]), str(overall_accus[2])]))
        con_log.write("=================================================")

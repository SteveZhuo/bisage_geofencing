from utils import *
import numpy as np
from sklearn.metrics import classification_report
from config import update_config
import time

n_bin = update_config["n_bin"]
parts = update_config["parts"]
dim = update_config["dim"]

contamination = update_config["contamination"]
temperature = update_config["temperature"]
threshold_t = update_config["threshold_l"]

def temperature_scaling(test_scores, t, threshold):
    prediction_ts = []
    test_score_ts = []
    for item in test_scores:
        sum = np.exp(item/t) + np.exp((1-item)/t)
        sigmoid_in_t = np.exp(item/t)/sum
        if sigmoid_in_t > threshold:
            label = 0
        else:
            label = 1
        test_score_ts.append(sigmoid_in_t)
        prediction_ts.append(label)
    prediction_ts = np.asarray(prediction_ts)
    return prediction_ts, test_score_ts


def process_results(files_id):
    if len(files_id) == 0:
        parse_items = exp_files
    else:
        parse_items = files_id
    result_dict = {}
    for file_id in parse_items:
        result_dict[file_id] = []
    return result_dict


def classify_with_sigmoid(files_id, result_dict, dim, part=10):
    for file_id in files_id:
        name = f'{update_config["output_dir_name"]}'
        print (f"file id: {file_id} \n")

        for idx in range(1, part+1):
            result = result_dict[file_id]
            scores_test = np.loadtxt(f'{name}/{file_id}/{idx}/score_{dim}.txt')
            labels_test = np.loadtxt(f'{name}/{file_id}/{idx}/pred_{dim}.txt')

            accuracy = []
            start_time = time.time()
            test_prediction_ts, test_score_ts = temperature_scaling(scores_test, temperature, threshold_t)
            label_test = labels_test
            report_test_ts = classification_report(label_test, test_prediction_ts, output_dict=True)
            print(f"Update for each node: {(time.time() - start_time) / len(test_prediction_ts)}")
            print(f"Process {idx} Done")
            print(report_test_ts)
            with open(f'{name}/{file_id}/{idx}/report.json', 'w') as f_in:
                json.dump(report_test_ts, f_in)

# Parameters
prune_num = 0
train = -1 # 1: prune in train set; 0: prune in update set
rep_size_global = 64
num_modify_global = 10

input_path = osp.join(os.getcwd(), update_config["emb_dir_name"])
exp_files = []
for filename in (os.listdir(input_path)):
    if not filename.startswith("v"): #vivo's data ignored for experiment this time
        exp_files.append(filename.split('_')[0])
files_id = [update_config["data_name"]]
result_dict = process_results(files_id)
classify_with_sigmoid(files_id, result_dict, dim, parts)
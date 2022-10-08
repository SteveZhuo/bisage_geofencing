import json
from numba.cuda import test
import numpy as np
from numpy.core.defchararray import equal
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from scipy.sparse import data
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
import sys
from utils import *
from data_loader import *
from config import update_config
import time

IN = 0
OUT = 1

class combined_od_model():
    def __init__(self, model_list):
        self.model_list = model_list
        self.model_num = len(model_list)
        self.majority = int(self.model_num/2)+1
    def predict(self, X):
        if len(X) == 0:
            return
        y = np.zeros(X.shape[0])
        predicted_scores = []
        for model in self.model_list:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    if np.isnan(X[i][j]):
                        X[i][j] = 0
            start_time = time.time()
            prediction = model.predict(X)
            # print(f"Prediction label: {prediction}")
            predicted_proba = model.predict_proba(X) # [a, b], a: inlier probability; b: 1-a
            # print(f"Prediction size: {len(predicted_proba)},  probability: {predicted_proba}")
            print(f"In-out detection for each node: {(time.time() - start_time) / X.shape[0]}")
            y += prediction
            predicted_scores += predicted_proba[:, 0].tolist()
        idx1 = y>=self.majority*OUT+(self.model_num-self.majority)*IN
        idx2 = y<=self.majority*IN+(self.model_num-self.majority)*OUT
        y[idx1] = OUT
        y[idx2] = IN
        return y, predicted_scores
        
def load_labels(filename, dat_filename):
    '''
    filename: of corresponding embedding file
    '''
    data_dict = load_dataset(dat_filename)
    id_list = []
    label_list = []
    with open(filename, "r") as f_in:
        lines = f_in.readlines()
        for line in lines:
            line = line.strip(" \n")
            entry_id, coors = line.split(" ", 1)
            id_list.append(int(entry_id[1:]))
    count = 0
    for _, label in data_dict['info']['label'].items():
        count += 1
        if count in id_list:
            label_list.append(label)
    return label_list

def load_embeddings(filename):
    embeddings = {}
    with open(filename, "r") as f_in:
        lines = f_in.readlines()
        for line in lines:
            line = line.strip(" \n")
            entry_id, coors = line.split(" ", 1)
            coor = [float(item) for item in coors.split(" ")]
            embeddings[entry_id] = coor
    return embeddings

def prep_embeddings(train_emb_filename, test_emb_filename, train_filename, test_filename, test_out_count, test_input_path, parts=10):
    parts_labels = []
    # train
    count = 0
    with open(train_emb_filename, "r") as f_in, open(train_filename, "w") as f_out:
        lines = f_in.readlines()
        for line in lines:
            if not line.startswith("u") and not line.startswith("i"):
                count = int(line.split(" ", 1)[0])
            elif not line.startswith("i"):
                f_out.write(line)
        f_in.close()
        f_out.close()

    # test
    test_count = 0
    with open(test_emb_filename, "r") as f_in:
        lines = f_in.readlines()
        for line in lines:
            if line.startswith("u"):
                test_count += 1

    part_ids = list(range(test_count))
    id_count = int(test_count/parts) # embedding for each part except the last

    part_inout_list = []
    with open(test_input_path, "r") as f_in:
        for line in f_in.readlines():
            if line.startswith("1"):
                part_inout_list.append(1)
            elif line.startswith("0"):
                part_inout_list.append(0)

    for part in range(parts):
        part_filename = test_filename.split(".txt")[0]+f"_{part+1}.txt"
        part_label = []
        cur_part_lines = 0
        cur_labels = []
        with open(test_emb_filename, "r") as f_in, open(part_filename, "w") as f_out:
            lines = f_in.readlines()
            cur_ids = part_ids[part*id_count:(part+1)*id_count] if part<parts-1 else part_ids[part*id_count:]
            for line in lines:
                if line.startswith("u"):
                    if cur_part_lines in cur_ids:
                        f_out.write(f"u{count+cur_part_lines+1} ")
                        f_out.write(line.split(" ", 1)[1])
                        part_label.append(part_inout_list[cur_part_lines])
                        # if cur_part_lines<=test_out_count:
                        #     part_label.append(1)
                        # else:
                        #     part_label.append(0)
                    cur_part_lines += 1
        parts_labels.append(part_label)

    return parts_labels


def build_od_model(X, n_bins):
    global contamination, current_model
    data_size = X.shape[0]
    hbos_model = HBOS(n_bins=n_bins, alpha=0.9, tol=5e-1, contamination=contamination).fit(X)
    """
    contamination_others = 1e-3
    fb_model = FeatureBagging(contamination=contamination_others).fit(X)
    lof_model = LOF(n_neighbors=10, contamination=contamination_others).fit(X)
    svdd_model = OCSVM(kernel="rbf", nu=0.05, gamma="auto").fit(X)
    iforest_model = IForest(contamination=contamination_others).fit(X)
    """

    return combined_od_model([hbos_model])


def check_update(data_folder, file_id, parts, rep_size, negative_ratio, offset, equal_flag, n_bins, name, dim, sigma, folder, parts_labels, **args):
    global contamination
    ratios = []
    if equal_flag:
        ratios = [idx/parts for idx in range(parts)] + [1.0]
    else:
        ratios = args["ratios"]

    temp = sys.stdout
    sys.stdout = Logger(os.path.join(data_folder, f'update_output/{file_id}', (f"log_update.txt")))

    raw_file = os.path.join(data_folder, "raw_input/{}.dat".format(file_id))

    embedding_id = file_id
    embedding_folder = os.path.join(data_folder) #, embedding_id, "temp")

    #print("initial process...")
    #print (f"bin size: {n_bins}")
    #print (f"file id: {file_id}")
    initial_embedding_file = os.path.join(os.getcwd(), f'update_output/{file_id}', f'dim_{dim}/train.txt')
    initial_embeddings = load_embeddings(initial_embedding_file)

    initial_X = np.array(list(initial_embeddings.values()))
    initial_y = np.zeros(initial_X.shape[0])
    labels = initial_y
    initial_od_model = build_od_model(initial_X, n_bins)
    #print("initial model trained.")

    groundtruth_ids = []
    for entry_id in initial_embeddings.keys():
        groundtruth_ids.append(entry_id)
    
    for idx in range(1, len(ratios)):
        print("testing process {}...".format(idx))

        update_embeddings = []
        update_length = 0
        update_y = []
        for update_idx in range(idx, len(ratios)):
            update_embedding_file = os.path.join(os.getcwd(), f'update_output/{file_id}', f'dim_{dim}/test_{update_idx}.txt')
            if update_idx==idx:
                update_embeddings = load_embeddings(update_embedding_file)
                update_length = len(update_embeddings)
                update_y = np.array(parts_labels[update_idx-1])
            else:
                i=0
                update_ys = np.array(parts_labels[update_idx-1])
                for key, value in load_embeddings(update_embedding_file).items():
                    if not key in update_embeddings.keys():
                        update_y = np.append(update_y, update_ys[i])
                        update_embeddings[key] = value
                    i+=1

        new_entry_ids = list(update_embeddings.keys())
        entry_ids = new_entry_ids

        update_X = np.array([update_embeddings[entry_id] for entry_id in new_entry_ids])

        update_prediction, update_score = initial_od_model.predict(update_X)

        # print result
        create_folder(os.path.join(os.getcwd(), name))
        create_folder(os.path.join(os.getcwd(), f'{name}/{file_id}'))
        create_folder(os.path.join(os.getcwd(), f'{name}/{file_id}/{idx}'))

        np.savetxt(f'{name}/{file_id}/{idx}/pred_{dim}.txt', update_prediction)
        np.savetxt(f'{name}/{file_id}/{idx}/score_{dim}.txt', update_score)

        #report_1 = classification_report(update_y, update_prediction, output_dict=True)
        #update_confuse = np.array2string(confusion_matrix(update_y, update_prediction))

        # increment initial_X and initial_y to train a new od_model
        for i in range(update_length):
            if update_score[i]<sigma:
                initial_X = np.vstack([initial_X, update_X[i]])
                initial_y = np.append(initial_y, update_y[i])
        start_time = time.time()
        initial_od_model = build_od_model(initial_X, n_bins)
        print(f"Update for each node: {(time.time() - start_time) / len(new_entry_ids)}")
    sys.stdout.end()
    sys.stdout = temp


def get_in_out_count(test_input_file):
    in_count, out_count = 0, 0
    with open(test_input_file, "r") as f_in:
        lines = f_in.readlines()
        for line in lines:
            if line.startswith("0"):
                in_count += 1
            elif line.startswith("1"):
                out_count += 1
    return in_count, out_count


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()

    def end(self):
        self.log.close()


parts = update_config["parts"] # parts to split test file into
rep_size = update_config["rep_size"]
negative_ratio = update_config["neg_ratio"]
offset = update_config["offset"]
equal_flag = update_config["equal_flag"]

n_bin = update_config["n_bin"]
contamination = update_config["contamination"]
dim = update_config["dim"]
sigma = update_config["threshold_u"]

data_name = update_config["data_name"]
output_folder = update_config["output_dir_name"]

if __name__ == "__main__":
    classification_report_test = []
    classification_report_all = []
    scores_test = []
    scores_all = []
    labels_test = []
    labels_all = []

    result_name_string = f'{output_folder}'
    input_path = osp.join(os.getcwd(), 'input')
    emb_path = osp.join(os.getcwd(), 'emb_output')
    train_out_path = osp.join(os.getcwd(), 'update_output')
    create_folder(train_out_path)

    train_input_path = osp.join(input_path, f"{data_name}_train", f'fingerprints_floor.txt')
    test_input_path = osp.join(input_path, f"{data_name}_test", f'fingerprints_floor.txt')
    train_emb_path = osp.join(emb_path, f"{data_name}_train", f'embedding/dim_{dim}.txt')
    test_emb_path = osp.join(emb_path, f"{data_name}_test", f'embedding/dim_{dim}.txt')

    test_in_count, test_out_count = get_in_out_count(test_input_path)#1500, 1251 # by getting number of lines from input folder
    train_path = osp.join(train_out_path, data_name)
    create_folder(train_path)
    train_dim_path = osp.join(train_path, f"dim_{dim}")
    create_folder(train_dim_path)
    train_file = os.path.join(train_dim_path, f'train.txt')
    test_file = os.path.join(train_dim_path, f'test.txt')
    parts_labels = prep_embeddings(train_emb_path, test_emb_path, train_file, test_file, test_out_count, test_input_path, parts=parts)

    check_update(os.getcwd(), data_name, parts, rep_size, negative_ratio, offset, equal_flag, n_bin, result_name_string, dim, sigma, output_folder, parts_labels)
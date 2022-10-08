import os
import os.path as osp
import json
import random
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
import shutil
from utils import *
from data_loader import *

IN = 0
OUT = 1

GROUND_TRUTH = 0
TEST=1

# contamination = 0.011
# contamination = 0.012 # prune-2-update
# contamination = 0.005 # prune-4-train
alpha_score = 0.012
# alpha_score = 0.9
# alpha_score = 0.5 # iForest

class combined_od_model():
    def __init__(self, model_list):
        self.model_list = model_list
        self.model_num = len(model_list)
        self.majority = int(self.model_num/2)+1
    def predict(self, X):
        y = np.zeros(X.shape[0])
        predicted_scores = []
        for model in self.model_list:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    if np.isnan(X[i][j]):
                        X[i][j] = 0
            prediction = model.predict(X)
            # print(f"Prediction label: {prediction}")
            predicted_proba = model.predict_proba(X) # [a, b], a: inlier probability; b: 1-a
            # print(f"Prediction size: {len(predicted_proba)},  probability: {predicted_proba}")
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

'''
def load_labels(filename):
    label_dict = {}
    with open(filename, "r", encoding="utf-8") as f_in:
        raw_data = json.load(f_in)
        for entry in raw_data["reports"]:
            entry_id = entry["dataId"]
            if "label" not in entry:
                continue
            label_dict[entry_id] = IN if entry["label"] == 1 else OUT

    return label_dict
'''

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

def split_embeddings(data_filename, emb_filename, train_filename, test_filename, train_ratio=0.9, parts=10):
    count = 0
    cur_line = 0
    with open(emb_filename, "r") as f_in, open(train_filename, "w") as f_out1, open(test_filename, "w") as f_out2:
        lines = f_in.readlines()
        for line in lines:
            if not line.startswith("u") and not line.startswith("i"):
                count, dim = line.split(" ", 1)
                ones_list = list(np.where(load_labels(emb_filename, data_filename))[0])
                num_train = int(int(count)*train_ratio)
                nums = num_train*[1] + (int(count)-num_train)*[0]
                random.shuffle(nums)
            elif not line.startswith("i"):
                if nums[cur_line]==1 and cur_line not in ones_list:
                    f_out1.write(line)
                else:
                    f_out2.write(line)
                cur_line += 1
        f_in.close()
        f_out1.close()
        f_out2.close()

    test_count = 0
    with open(test_filename, "r") as f_in:
        lines = f_in.readlines()
        test_count = len(lines)

    part_ids = list(range(test_count))
    random.shuffle(part_ids)
    id_count = int(test_count/parts) # embedding for each part

    for part in range(parts):
        part_filename = test_filename.split(".txt")[0]+f"_{part+1}.txt"
        cur_line = 0
        with open(test_filename, "r") as f_in, open(part_filename, "w") as f_out:
            lines = f_in.readlines()
            cur_ids = part_ids[part*id_count:(part+1)*id_count]
            for line in lines:
                if cur_line in cur_ids:
                    f_out.write(line)
                cur_line += 1
            f_in.close()
            f_out.close()


def find_new_embeddings(old_embeddings, new_embeddings):
    diff_entries = list(set(new_embeddings.keys()).difference(set(old_embeddings.keys())))
    same_entries = list(set(new_embeddings.keys()).intersection(set(old_embeddings.keys())))
    #print(f"old: {len(old_embeddings)}, new: {len(new_embeddings)}, added: {len(diff_entries)}, overlap: {len(same_entries)}")
    return {entry_id:new_embeddings[entry_id] for entry_id in diff_entries}, {entry_id:new_embeddings.get(entry_id,old_embeddings[entry_id]) for entry_id in old_embeddings.keys()}


def build_od_model(X, n_bins):
    global contamination, current_model
    data_size = X.shape[0]
    # if data_size <= 100:
    #     contamination = min(1e-1 / data_size, 1e-3)
    # elif 100 < data_size <= 200:
    #     contamination = 3 / data_size
    # elif 200 < data_size <= 300:
    #     contamination = 5 / data_size
    # else:
    #     contamination = 8 / data_size
    # contamination = max(float(1)/float(data_size), 1e-3)
    #print (f"Contamination: {contamination}")
    # hbos_model = HBOS(n_bins=18, alpha=0.9, tol=5e-1, contamination=contamination).fit(X)
    hbos_model = HBOS(n_bins=n_bins, alpha=0.9, tol=5e-1, contamination=contamination).fit(X)
    contamination_others = 1e-3
    fb_model = FeatureBagging(contamination=contamination_others).fit(X)
    lof_model = LOF(n_neighbors=10, contamination=contamination_others).fit(X)
    svdd_model = OCSVM(kernel="rbf", nu=0.05, gamma="auto").fit(X)
    iforest_model = IForest(contamination=contamination_others).fit(X)

    # if current_model == 'svdd':
    #     return combined_od_model([svdd_model])
    # if current_model == 'iforest':
    #     return combined_od_model([iforest_model])
    # if current_model == 'lof':
    #     return combined_od_model([lof_model])
    # if current_model == 'fb':
    #     return combined_od_model([fb_model])
    # return combined_od_model([hbos_model, svdd_model, iforest_model, lof_model, fb_model])
    return combined_od_model([hbos_model])




def od_update(new_embeddings, old_embeddings, label_dict, prediction_dict, score_dict, groundtruth_ids, initial_flag, n_bins):
    FLAG_IN_DATA_ONLY = True
    training_embeddings = old_embeddings.copy()
    training_embeddings.update(new_embeddings)
    training_entry_ids = list(training_embeddings.keys())

    # Prediction: in -> 80% used for update.
    # Prediction: out -> ignored.
    def od_update_0():
        chosen_entry_ids = []
        for entry_id in training_embeddings:
            if prediction_dict[entry_id] == IN and random.random()<=0.8:
                chosen_entry_ids.append(entry_id)
        return chosen_entry_ids

    # 40% -> 70% data can be used for update.
    # Only in data are considered.
    def od_update_1():
        th1 = 0.6
        th2 = 0.01
        while True:
            chosen_entry_ids = []
            for entry_id in training_embeddings:
                if prediction_dict[entry_id] == IN and random.random()<=th1:
                    chosen_entry_ids.append(entry_id)
                if prediction_dict[entry_id] == OUT and random.random()<= th2:
                    chosen_entry_ids.append(entry_id)
            if 0.4 * len(training_embeddings) > len(chosen_entry_ids):
                th1 -= 0.02
                if th1 <= 0.5:
                    th1 = 0.5
                th2 += 0.005
                if th2 >= 0.5:
                    th2 = 0.5
            elif len(chosen_entry_ids) > 0.7 * len(training_embeddings):
                th1 -= 0.01
                th2 -= 0.005
                if th2 <= 0.01:
                    th2 = 0.01
                if th1 <= 0.5:
                    th1 = 0.5
            else:
                break
        return chosen_entry_ids

    # Prediction: in; selected with threshold_1.
    # Retrain and test. If results are consistent, put the new data in. If not, put in with probability.
    def od_update_2():
        #print("baseline:\n", classification_report([label_dict[entry_id] for entry_id in training_entry_ids], [prediction_dict[entry_id] for entry_id in training_entry_ids]))
        th1, th2, th3 = 0.0, 0.0, 0.0
        if initial_flag:
            th1 = 0.8
            th2 = 0.5
            th3 = 0.05
        else:
            th1 = 0.9
            th2 = 0.6
            th3 = 0.01

        if initial_flag:
            chosen_entry_ids = set(old_embeddings.keys())
        else:
            chosen_entry_ids = set()
        for entry_id in training_embeddings:
            if prediction_dict[entry_id] == IN and random.random()<=th1:
                chosen_entry_ids.add(entry_id)

        X = np.array([training_embeddings[entry_id] for entry_id in list(chosen_entry_ids)])
        model = build_od_model(X, n_bins)
        test_X = np.array([training_embeddings[entry_id] for entry_id in training_entry_ids])
        test_y = np.array(load_labels(test_embedding_file, raw_file))
        test_prediction = model.predict(test_X)
        # print(f"Decision scores: size: {len(hbos_model.decision_scores_)}, scores: {hbos_model.decision_scores_}")
        new_prediction_dict = {}
        for idx, entry_id in enumerate(training_entry_ids):
            new_prediction_dict[entry_id] = test_prediction[idx]
        #print("update 1:\n", classification_report([label_dict[entry_id] for entry_id in training_entry_ids], [new_prediction_dict[entry_id] for entry_id in training_entry_ids]))

        if initial_flag:
            new_chosen_entry_ids = set(old_embeddings.keys())
        else:
            new_chosen_entry_ids = set()
        for times in range(1):
            if initial_flag:
                new_chosen_entry_ids = set(old_embeddings.keys())
            else:
                new_chosen_entry_ids = set()
            for entry_id in training_embeddings:
                if new_prediction_dict[entry_id] == IN and prediction_dict[entry_id] == new_prediction_dict[entry_id]:
                    new_chosen_entry_ids.add(entry_id)
                elif new_prediction_dict[entry_id] == IN and random.random()<=th2:
                    new_chosen_entry_ids.add(entry_id)
                elif new_prediction_dict[entry_id] == OUT and random.random()<=th3:
                    new_chosen_entry_ids.add(entry_id)
            X2 = np.array([training_embeddings[entry_id] for entry_id in list(new_chosen_entry_ids)])
            model2 = build_od_model(X2, n_bins)
            test_X2 = np.array([training_embeddings[entry_id] for entry_id in training_entry_ids])
            test_prediction2 = model2.predict(test_X2)
            new_prediction_dict2 = {}
            for idx, entry_id in enumerate(training_entry_ids):
                new_prediction_dict2[entry_id] = test_prediction2[idx]
            #print("update 2:\n", classification_report([label_dict[entry_id] for entry_id in training_entry_ids], [new_prediction_dict2[entry_id] for entry_id in training_entry_ids]))
        
        return list(new_chosen_entry_ids)

    # Confidence: positve prediction to be correct.
    # Correction: negative prediction should be corrected.
    # Choose from prediction data with ratio of confidence/correction to use as in-boundary signals.
    def od_update_3():
        p_ids = [entry_id for entry_id in training_embeddings if prediction_dict[entry_id] == IN]
        n_ids = [entry_id for entry_id in training_embeddings if prediction_dict[entry_id] == OUT]
        p_ratio = float(len(p_ids) / (len(p_ids) + len(n_ids)))
        data_size = len(training_embeddings)
        confidence = 0.6*p_ratio + 0.5 if p_ratio <= 2/3 else -1.2*p_ratio + 1.7
        correction = 0.3/(1+np.e**(-p_ratio))

        confidence *= 1 - np.e**(-data_size/100)
        correction *= 1 - np.e**(-data_size/100)

        if initial_flag == True:
            chosen_entry_ids = set(old_embeddings.keys())
        else:
            chosen_entry_ids = set()

        p_entry_ids = [entry_id for entry_id in new_embeddings if prediction_dict[entry_id] == IN]
        n_entry_ids = [entry_id for entry_id in new_embeddings if prediction_dict[entry_id] == OUT]
        p_num = max(20, int(confidence*len(p_entry_ids)))
        n_num = max(1, int(correction*len(n_entry_ids)))
        if len(p_entry_ids) == 0:
            p_selection = set()
            #print("warning: no positive data")
        else:
            p_selection = {p_entry_ids[idx] for idx in np.random.choice(len(p_entry_ids), p_num)}
        if len(n_entry_ids) == 0:
            n_selection = set()
        else:
            n_selection = {n_entry_ids[idx] for idx in np.random.choice(len(n_entry_ids), n_num)}
        chosen_entry_ids.update(p_selection)
        chosen_entry_ids.update(n_selection)

        return chosen_entry_ids

    # Get in-boundary data whose prediction score higher than a threshold.
    def od_update_6():
        min_score = 1
        max_score = 0

        # threshold_score = alpha_score * contamination * len(groundtruth_ids)
        threshold_score = alpha_score
        #print (f"Threshold_score is : {threshold_score}")
        chosen_entry_ids = []
        for entry_id in groundtruth_ids:
            if score_dict[entry_id] < min_score:
                min_score = score_dict[entry_id]
            if score_dict[entry_id] > max_score:
                max_score = score_dict[entry_id]
        #print(f"max score: {max_score}; min score: {min_score}")
        num_new = 0
        for entry_id in training_embeddings:
            if prediction_dict[entry_id] == IN:
                if entry_id in groundtruth_ids:
                    chosen_entry_ids.append(entry_id)
                else:
                    if score_dict[entry_id] >= (max_score - min_score) * threshold_score:
                        # if random.random() <= score_dict[entry_id]:
                            # print (f"Choosen score: {score_dict[entry_id]}")
                        chosen_entry_ids.append(entry_id)
                        groundtruth_ids.append(entry_id)
                        num_new += 1
        #print (f"Newly added data number: {num_new}")
        return chosen_entry_ids
    """
    Following schemes consider both inliers and outliers.
    """
    # Similar to od_update_0; random dropout rate (20%)
    def od_update_4():
        chosen_entry_ids = []
        # Choose from predicted signals.
        for entry_id in prediction_dict:
            if random.random() <= 0.8:
                chosen_entry_ids.append(entry_id)
        return chosen_entry_ids

    # Similar to od_update_1.
    def od_update_5():
        chosen_entry_ids = []
        th1 = 0.6
        th2 = 0.01
        while True:
            chosen_entry_ids = []
            for entry_id in prediction_dict:
                if prediction_dict[entry_id] == IN and random.random() <= th1:
                    chosen_entry_ids.append(entry_id)
                if prediction_dict[entry_id] == OUT and random.random() <= th2:
                    chosen_entry_ids.append(entry_id)
            if 0.4 * len(prediction_dict) > len(chosen_entry_ids):
                th1 -= 0.02
                if th1 <= 0.5:
                    th1 = 0.5
                th2 += 0.005
                if th2 >= 0.5:
                    th2 = 0.5
            elif len(chosen_entry_ids) > 0.7 * len(prediction_dict):
                th1 -= 0.01
                th2 -= 0.005
                if th2 <= 0.01:
                    th2 = 0.01
                if th1 <= 0.5:
                    th1 = 0.5
            else:
                break
        return chosen_entry_ids

    chosen_entry_ids = od_update_6() #TODO:change this line to customize


    num_out = 0
    if FLAG_IN_DATA_ONLY:
        # Keep only in data, for od_update methods 0,1,2,3
        for entry_id in chosen_entry_ids:
            prediction_dict[entry_id] = IN
    else:
        for entry_id in chosen_entry_ids:
            if prediction_dict[entry_id] == OUT:
                num_out += 1

        # Choose from original signals.
        for entry_id in training_embeddings:
            if random.random() <= 0.8 and entry_id not in prediction_dict:
                chosen_entry_ids.append(entry_id)
        # contamination = float(num_out)/float(len(chosen_entry_ids))

    # gtc_list = [int(label_dict[entry_id]) for entry_id in chosen_entry_ids]
    # print("ground truth for chosen points")
    # print(gtc_list)

    # pc_list = [int(prediction_dict[entry_id]) for entry_id in chosen_entry_ids]
    # print("predictions for chosen points")
    # print(pc_list)

    # print("for chosen points:\n", classification_report(gtc_list, pc_list))

    # gtn_list = [int(label_dict[entry_id]) for entry_id in training_embeddings if entry_id not in chosen_entry_ids]
    # print("ground truth for not chosen points")
    # print(gtn_list)

    # pn_list = [int(prediction_dict[entry_id]) for entry_id in training_embeddings if entry_id not in chosen_entry_ids]
    # print("predictions for not chosen points")
    # print(pn_list)

    # print("for not chosen points:\n", classification_report(gtn_list, pn_list))

    # Use chosen embeddings to learn boundary.
    X = np.array([training_embeddings[entry_id] for entry_id in chosen_entry_ids])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.isnan(X[i][j]):
                X[i][j] = 0
    model = build_od_model(X, n_bins)

    # Predict labels for all embeddings.
    # This is to "correct" possibly mislabelled data assuming that new model is better than old one.
    # X2 = np.array([training_embeddings[entry_id] for entry_id in training_entry_ids])
    # prediction = model.predict(X2)

    # Referenced variable. Updated after each batch.
    # prediction_dict = {}
    # for idx, entry_id in enumerate(training_entry_ids):
    #     prediction_dict[entry_id] = prediction[idx]
    # print("performance after update:\n", classification_report([label_dict[entry_id] for entry_id in training_entry_ids], [prediction_dict[entry_id] for entry_id in training_entry_ids]))
    return model


# Return predicted labels and ground truth labels.
def test_files(raw_file, embeddings, X, y, od_model, labels, idx, ratios, parts, rep_size, negative_ratio, offset, embedding_folder):
    global classification_report_test, classification_report_all, scores_test, labels_test, scores_all, labels_all, temperature, threshold_t
    test_ys = np.array([]) # True labels.
    test_predictions = np.array([]) # Predicted labels.
    test_scores = np.array([]) # Predicted scores.
    # test_predictions_ts = np.array([])
    # test_scores_ts = np.array([])
    entry_ids = []
    current_id = idx

    # Prediction for all.
    for ratio in ratios[idx:len(ratios)-1]:
        test_embedding_file = os.path.join(embedding_folder, f"test_dim_{dim}.txt")
        #test_embedding_file = os.path.join(embedding_folder, "embeddings_predict{}-{}_test{}-{}-{}_{}_{}_{}.txt".format(parts, ratios[idx], parts, ratios[idx], ratio, rep_size, negative_ratio, offset))
        test_embeddings = load_embeddings(test_embedding_file)

        # Find embeddings inserted at this round of test/update.
        test_new_embeddings, _ = find_new_embeddings(embeddings, test_embeddings)

        new_entry_ids = list(test_new_embeddings.keys())
        entry_ids = entry_ids + new_entry_ids
        
        test_y = np.array(load_labels(test_embedding_file, raw_file))

        #test_y = np.array([labels[entry_id] for entry_id in new_entry_ids])
        test_X = np.array([test_new_embeddings[entry_id] for entry_id in new_entry_ids])

        test_prediction, test_score = od_model.predict(test_X)

        # test_prediction_ts, test_score_ts = temperature_scaling(test_score, temperature, threshold_t)
        # print("testing original: \n", classification_report(test_y, test_prediction))
        # print("testing temperature scaling: \n", classification_report(test_y, test_prediction_ts))
        # Final batch for testing
        # if current_id == len(ratios) -2:
        #     print("testing: \n", classification_report(test_y, test_prediction))

        test_ys = np.concatenate((test_ys, test_y), axis=0)
        test_predictions = np.concatenate((test_predictions, test_prediction), axis=0)
        test_scores = np.concatenate((test_scores, test_score), axis=0)
        # test_predictions_ts = np.concatenate((test_predictions_ts, test_prediction_ts), axis=0)
        # test_scores_ts = np.concatenate((test_scores_ts, test_score_ts), axis=0)

        current_id += 1

    report_1 = classification_report(test_ys, test_predictions, output_dict=True)
    #print("testing total: \n", report_1)
    classification_report_test.append(report_1)
    scores_test.append(test_scores.tolist())
    labels_test.append(test_ys.tolist())

    # report_1_ts = classification_report(test_ys, test_predictions_ts, output_dict=True)
    # print("testing total ts: \n", report_1_ts)
    # classification_report_test_ts.append(report_1_ts)

    test_num = len(entry_ids)
    entry_ids = entry_ids + list(embeddings.keys())

    prediction_original, score_original = od_model.predict(X)
    # prediction_original_ts, score_original_ts = temperature_scaling(score_original, temperature, threshold_t)
    test_ys = np.concatenate((test_ys, y), axis=0)
    test_predictions_all = np.concatenate((test_predictions, prediction_original), axis=0)
    test_scores_all = np.concatenate((test_scores, score_original), axis=0)
    # test_predictions_all_ts = np.concatenate((test_predictions_ts, prediction_original_ts), axis=0)
    # test_scores_all_ts = np.concatenate((test_scores_ts, score_original_ts), axis=0)

    report_2 = classification_report(test_ys, test_predictions_all, output_dict=True)
    #print("all total: \n", report_2)
    classification_report_all.append(report_2)
    scores_all.append(test_scores_all.tolist())
    labels_all.append(test_ys.tolist())

    # report_2_ts = classification_report(test_ys, test_predictions_all_ts, output_dict=True)
    # print("all total: \n", report_2_ts)
    # classification_report_all_ts.append(report_2_ts)

    prediction_dict = {}
    y_dict = {}
    score_dict = {}
    for idx, entry_id in enumerate(entry_ids):
        y_dict[entry_id] = test_ys[idx]
        prediction_dict[entry_id] = test_predictions_all[idx]
        score_dict[entry_id] = test_scores_all[idx]
    return y_dict, prediction_dict, score_dict


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


def check_update(data_folder, file_id, parts, rep_size, negative_ratio, offset, equal_flag, n_bins, ratio, name, dim, sigma, **args):
    global contamination
    ratios = []
    if equal_flag:
        ratios = [idx/parts for idx in range(parts)] + [1.0]
    else:
        ratios = args["ratios"]

    temp = sys.stdout
    sys.stdout = Logger(os.path.join(data_folder, f'update_output/{file_id}', (f"log_update_{contamination}.txt")))

    raw_file = os.path.join(data_folder, "raw_input/{}.dat".format(file_id))
    #labels = load_labels(raw_file)

    embedding_id = file_id
    embedding_folder = os.path.join(data_folder) #, embedding_id, "temp")

    #print("initial process...")
    #print (f"bin size: {n_bins}")
    #print (f"file id: {file_id}")
    initial_embedding_file = os.path.join(os.getcwd(), f'update_output/{file_id}', f'dim_{dim}_ratio_{ratio}/train.txt')
    #initial_embedding_file = os.path.join(embedding_folder, "embeddings_initial{}_initial_{}_{}_{}.txt".format(parts, rep_size, negative_ratio, offset))
    initial_embeddings = load_embeddings(initial_embedding_file)

    initial_X = np.array(list(initial_embeddings.values()))
    initial_y = np.array(load_labels(initial_embedding_file, raw_file))
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
            update_embedding_file = os.path.join(os.getcwd(), f'update_output/{file_id}', f'dim_{dim}_ratio_{ratio}/test_{update_idx}.txt')
            if update_idx==idx:
                update_embeddings = load_embeddings(update_embedding_file)
                update_length = len(update_embeddings)
                update_y = np.array(load_labels(update_embedding_file, raw_file))
            else:
                i=0
                update_ys = np.array(load_labels(update_embedding_file, raw_file))
                for key, value in load_embeddings(update_embedding_file).items():
                    if not key in update_embeddings.keys():
                        update_y = np.append(update_y, update_ys[i])
                        update_embeddings[key] = value
                    i+=1
                #update_y = np.append(update_y, np.array(load_labels(update_embedding_file, raw_file)))
                #update_embeddings.update(load_embeddings(update_embedding_file))

        new_entry_ids = list(update_embeddings.keys())
        entry_ids = new_entry_ids

        update_X = np.array([update_embeddings[entry_id] for entry_id in new_entry_ids])
        update_prediction, update_score = initial_od_model.predict(update_X)

        # print result
        if not os.path.isdir(os.path.join(os.getcwd(), 'result_iters')):
            os.mkdir(os.path.join(os.getcwd(), 'result_iters'))
        if not os.path.isdir(os.path.join(os.getcwd(), name)):
            os.mkdir(os.path.join(os.getcwd(), name))
        if not os.path.isdir(os.path.join(os.getcwd(), f'{name}/{ratio}')):
            os.mkdir(os.path.join(os.getcwd(), f'{name}/{ratio}'))
        if not os.path.isdir(os.path.join(os.getcwd(), f'{name}/{ratio}/{file_id}')):
            os.mkdir(os.path.join(os.getcwd(), f'{name}/{ratio}/{file_id}'))
        if not os.path.isdir(os.path.join(os.getcwd(), f'{name}/{ratio}/{file_id}/{idx}')):
            os.mkdir(os.path.join(os.getcwd(), f'{name}/{ratio}/{file_id}/{idx}'))
        if not os.path.isdir(os.path.join(os.getcwd(), f'{name}/{ratio}/{file_id}/{idx}/{sigma}')):
            os.mkdir(os.path.join(os.getcwd(), f'{name}/{ratio}/{file_id}/{idx}/{sigma}'))
        #with open(f'{name}/{ratio}/{file_id}/{idx}/pred_{dim}.txt', 'w') as f_o1, open(f'{name}/{ratio}/{file_id}/{idx}/score_{dim}.txt', 'w') as f_o2:
        #    f_o1.close()
        #    f_o2.close()

        np.savetxt(f'{name}/{ratio}/{file_id}/{idx}/pred_{dim}.txt', update_prediction)
        np.savetxt(f'{name}/{ratio}/{file_id}/{idx}/score_{dim}.txt', update_score)

        report_1 = classification_report(update_y, update_prediction, output_dict=True)
        update_confuse = np.array2string(confusion_matrix(update_y, update_prediction))
        with open(f'{name}/{ratio}/{file_id}/{idx}/report_no_scale_dim_{dim}.json', 'w') as f_in:
            json.dump(report_1, f_in)

        # increment initial_X and initial_y to train a new od_model
        for i in range(update_length):
            if update_score[i]<sigma:
                initial_X = np.vstack([initial_X, update_X[i]])
                initial_y = np.append(initial_y, update_y[i])
        initial_od_model = build_od_model(initial_X, n_bins)
    '''
    test_embedding_file = os.path.join(embedding_folder, f'train_output/{file_id}', f"test_dim_{dim}_ratio_{ratio}.txt")
    test_embeddings = load_embeddings(test_embedding_file)

    new_entry_ids = list(test_embeddings.keys())
    entry_ids = new_entry_ids
    test_y = np.array(load_labels(test_embedding_file, raw_file))

    #test_y = np.array([labels[entry_id] for entry_id in new_entry_ids])
    test_X = np.array([test_embeddings[entry_id] for entry_id in new_entry_ids])
    test_prediction, test_score = initial_od_model.predict(test_X)
    
    if not os.path.isdir(os.path.join(os.getcwd(), 'result')):
        os.mkdir(os.path.join(os.getcwd(), 'result'))
    if not os.path.isdir(os.path.join(os.getcwd(), name)):
        os.mkdir(os.path.join(os.getcwd(), name))
    if not os.path.isdir(os.path.join(os.getcwd(), f'{name}/{ratio}')):
        os.mkdir(os.path.join(os.getcwd(), f'{name}/{ratio}'))
    if not os.path.isdir(os.path.join(os.getcwd(), f'{name}/{ratio}/{file_id}')):
        os.mkdir(os.path.join(os.getcwd(), f'{name}/{ratio}/{file_id}'))
    with open(f'{name}/{ratio}/{file_id}/pred_{dim}.txt', 'w') as f_o1, open(f'{name}/{ratio}/{file_id}/score_{dim}.txt', 'w') as f_o2:
        f_o1.close()
        f_o2.close()

    np.savetxt(f'{name}/{ratio}/{file_id}/pred_{dim}.txt', test_prediction)
    np.savetxt(f'{name}/{ratio}/{file_id}/score_{dim}.txt', test_score)
    #print(test_y)
    #print(test_prediction)

    report_1 = classification_report(test_y, test_prediction, output_dict=True)
    test_confuse = np.array2string(confusion_matrix(test_y, test_prediction))
    #print("testing total: \n", report_1)
    #print("testing total: \n", test_confuse)
    with open(f'{name}/{ratio}/{file_id}/report_no_scale_dim_{dim}.json', 'w') as f_in:
        json.dump(report_1, f_in)
        '''

    # for idx in range(1, len(ratios)-1):
    #     print("testing process {}...".format(idx))

    #     # y_dict: ground truth labels.
    #     # prediction_dict: predicted labels.
    #     # score_dict: predicted scores.
    #     # category_dict: entry belongs to ground truth or testing data
    #     # key: entry_id
    #     y_dict, prediction_dict, score_dict = test_files(raw_file, initial_embeddings, initial_X, initial_y, initial_od_model, labels, idx, ratios, parts, rep_size, negative_ratio, offset, embedding_folder)

    #     print("updating process {}...".format(idx))

    #     updated_embedding_file = os.path.join(embedding_folder, "train_dim_32.txt")
    #     #updated_embedding_file = os.path.join(embedding_folder, "embeddings_update{}-{}_update{}-{}_{}_{}_{}.txt".format(parts, ratios[idx], parts, ratios[idx], rep_size, negative_ratio, offset))
    #     updated_embeddings = load_embeddings(updated_embedding_file)
    #     updated_new_embeddings, updated_old_embeddings = find_new_embeddings(initial_embeddings, updated_embeddings)

    #     initial_embeddings = updated_embeddings
    #     initial_X = np.array(list(initial_embeddings.values()))
    #     initial_y = np.array(load_labels(initial_embedding_file, raw_file))
    #     #initial_y = np.array([labels[entry_id] for entry_id in initial_embeddings])
    #     initial_flag = True if idx == 1 else False
    #     initial_od_model = od_update(updated_new_embeddings, updated_old_embeddings, y_dict, prediction_dict, score_dict, groundtruth_ids, initial_flag, n_bins)

    #     print("models and embeddings updated.")
    sys.stdout.end()
    sys.stdout = temp


# {'label 1': {'precision':0.5,
#                          'recall':1.0,
#                          'f1-score':0.67,
#                          'support':1},
# 'label 2': { ... },
#               ...
# }
def plot_result(reports, tested=True):
    global folder, n_bins, file_id, contamination
    f_score_in = []
    f_score_out = []
    for item in reports:
        if '0.0' and '1.0' in item.keys():
            f_score_in.append(item['0.0']['f1-score'])
            f_score_out.append(item['1.0']['f1-score'])
    x_interval = range(len(f_score_in))
    fontsize = 20
    fig, ax = plt.subplots()
    ax.plot(x_interval, f_score_in, '^-', markersize=10, linewidth=2, label="F-score (in)", color='royalblue')
    ax.plot(x_interval, f_score_out, 'o-', markersize=10, linewidth=2, label="F-score (out)", color='r')
    plt.xlabel("Ratio of data used (%)", fontsize=fontsize)
    plt.ylabel("Score", fontsize=fontsize)
    xticks = range(10, 100, 10)
    plt.xticks(range(9), xticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim([10, 90])
    plt.ylim([0, 1])
    legend = ax.legend(fontsize=fontsize)
    create_folder(f"{folder}/results-contamination-{contamination}")
    plt.tight_layout()
    if tested:
        plt.savefig(f"{folder}/results-contamination-{contamination}/id_{file_id}_bin_{n_bins}_tested.png", dpi=1000)
    else:
        plt.savefig(f"{folder}/results-contamination-{contamination}/id_{file_id}_bin_{n_bins}_all.png", dpi=1000)
    plt.close()
    print(f"Results plotted for id: {file_id} bin: {n_bins}")


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()  # 每次写入后刷新到文件中，防止程序意外结束

    def flush(self):
        self.log.flush()

    def end(self):
        self.log.close()


def move_and_rename_dirs():
    root_folder = os.getcwd()
    data_folder = os.path.join(root_folder, "data")
    data_folder_pruned = os.path.join(root_folder, "data-pruned")

    for rep_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        # Make dirs for different rep sizes.
        path = os.path.join(root_folder, f"update-rep-{rep_size}-1-10")
        try:
            os.mkdir(path)
            # Copy .dat files to corresponding dirs.
            for sub_path in os.listdir(data_folder):
                if sub_path.endswith(".dat"):
                    file_from = os.path.join(data_folder, sub_path)
                    file_to = os.path.join(path, sub_path)
                    shutil.copyfile(file_from, file_to)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

    # Copy data dirs for corresponding rep size and rename.
    for sub_path in os.listdir(data_folder):
        items = sub_path.split("-")
        if len(items) > 4:
            if items[-4] == "rep":
                path_from = os.path.join(data_folder, sub_path)
                path_to = os.path.join(root_folder, f"update-rep-{items[-3]}-1-10")
                shutil.move(path_from, path_to)
                folder_name_old = os.path.join(path_to, sub_path)
                folder_name_new = os.path.join(path_to, items[0])
                os.rename(folder_name_old, folder_name_new)

    for prune_num in [1,2,3,4,5]:
        # Make dirs for pruned APs in train set.
        path = os.path.join(root_folder, f"update-rep-64-1-10-pruned-{prune_num}-train")
        try:
            os.mkdir(path)
            for sub_path in os.listdir(data_folder_pruned):
                if sub_path.endswith(".dat"):
                    file_from = os.path.join(data_folder_pruned, sub_path)
                    file_to = os.path.join(path, sub_path)
                    shutil.copyfile(file_from, file_to)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

        # Make dirs for pruned APs in update set.
        path = os.path.join(root_folder, f"update-rep-64-1-10-pruned-{prune_num}-update")
        try:
            os.mkdir(path)
            for sub_path in os.listdir(data_folder_pruned):
                if sub_path.endswith(".dat"):
                    file_from = os.path.join(data_folder_pruned, sub_path)
                    file_to = os.path.join(path, sub_path)
                    shutil.copyfile(file_from, file_to)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)
    # Copy data dirs to corresponding prune dir and rename.
    for sub_path in os.listdir(data_folder_pruned):
        items = sub_path.split("-")
        if len(items) >= 4:
            if items[-3] == "prune":
                path_from = os.path.join(data_folder_pruned, sub_path)
                path_to = os.path.join(root_folder, f"update-rep-64-1-10-pruned-{items[-2]}-{items[-1]}")
                shutil.move(path_from, path_to)
                folder_name_old = os.path.join(path_to, sub_path)
                folder_name_new = os.path.join(path_to, items[0])
                os.rename(folder_name_old, folder_name_new)


# move_and_rename_dirs()

parts = 10 # parts to split test file into
rep_size = 16
negative_ratio = 4
offset = 120
equal_flag = True
# params for [0.6, ..., 1]
# temperature = 0.05
# threshold_t = 0.002

# temperature = 0.01
# threshold_t = 0.0005

if __name__ == "__main__":
    n_bins = [16]#list(range(5,20))#[8, 16, 32]
    contaminations = [0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 0.013, 0.015, 0.017, 0.019, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.2, 0.3, 0.4, 0.5]
    ratios = [0.9]#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dims = [32]
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    classification_report_test = []
    classification_report_all = []
    scores_test = []
    scores_all = []
    labels_test = []
    labels_all = []

    for n_bin in n_bins:
        for contamination in contaminations:
            for dim in dims:
                result_name_string = f'result_iters/{n_bin}_{contamination}'
                for ratio in ratios:
                    input_path = osp.join(os.getcwd(), 'raw_input')
                    emb_path = osp.join(os.getcwd(), 'emb_output')
                    train_out_path = osp.join(os.getcwd(), 'update_output')
                    if not os.path.isdir(train_out_path):
                        os.mkdir(train_out_path)
                    for filename in (os.listdir(input_path)):
                        if not filename.startswith("v"):
                            data_path = osp.join(input_path, filename) # .dat file
                            data_name = filename.split('.')[0]
                            
                            emb_file_path = osp.join(emb_path, data_name, f'embedding/dim_{dim}.txt')
                            train_path = osp.join(train_out_path, data_name)
                            if not os.path.isdir(train_path):
                                os.mkdir(train_path)
                            train_dim_path = osp.join(train_path, f"dim_{dim}_ratio_{ratio}")
                            if not os.path.isdir(train_dim_path):
                                os.mkdir(train_dim_path)
                            train_file = os.path.join(train_dim_path, f'train.txt')
                            test_file = os.path.join(train_dim_path, f'test.txt')
                            split_embeddings(data_path, emb_file_path, train_file, test_file, train_ratio=ratio, parts=parts)
                    for filename in (os.listdir(input_path)):
                        for sigma in sigmas:
                            if not filename.startswith("v"):
                                data_path = osp.join(input_path, filename) # .dat file
                                data_name = filename.split('.')[0]
                                check_update(os.getcwd(), data_name, parts, rep_size, negative_ratio, offset, equal_flag, n_bin, ratio, result_name_string, dim, sigma)
            print(f"bin_{n_bin}_cont_{contamination} done.")

'''
exp_files = [
    '2g.dat',
    '5g.dat',
    # 'accuracy_heatmap_1.dat',
    # 'accuracy_heatmap_2.dat',
    # 'accuracy_heatmap_3.dat',
    # 'accuracy_heatmap_4.dat',
    # 'accuracy_heatmap_5.dat',
    # 'accuracy_heatmap_6.dat',
    # 'accuracy_heatmap_7.dat',
    # 'accuracy_heatmap_8.dat',
    # 'accuracy_heatmap_9.dat',
    # 'accuracy_heatmap_10.dat',
    # 'accuracy_heatmap_11.dat',
    # 'accuracy_heatmap_12.dat',
    # 'accuracy_heatmap_13.dat',
    # 'accuracy_heatmap_14.dat',
    # 'accuracy_heatmap_15.dat',
    # 'accuracy_heatmap_16.dat',
    # 'accuracy_heatmap_17.dat',
    # 'accuracy_heatmap_18.dat',
    # 'accuracy_heatmap_19.dat',
    # 'accuracy_heatmap_20.dat',
    # 'accuracy_heatmap_21.dat',
    # 'accuracy_heatmap_22.dat',
    # 'accuracy_heatmap_23.dat',
    # 'accuracy_heatmap_24.dat',
    # 'accuracy_heatmap_25.dat',
    # 'accuracy_heatmap_26.dat',
    # 'accuracy_heatmap_27.dat',
    # 'accuracy_heatmap_28.dat',
    # 'accuracy_heatmap_29.dat',
    # 'accuracy_heatmap_30.dat',
    # 'accuracy_heatmap_31.dat',
    # 'accuracy_heatmap_32.dat',
    # 'accuracy_heatmap_33.dat',
    # 'accuracy_heatmap_34.dat',
    # 'accuracy_heatmap_35.dat',
    # 'accuracy_heatmap_36.dat',
    # 'accuracy_heatmap_37.dat',
    # 'accuracy_heatmap_38.dat',
    # 'accuracy_heatmap_39.dat',
    # 'accuracy_heatmap_40.dat',
    # 'accuracy_heatmap_41.dat',
    # 'accuracy_heatmap_42.dat',
    # 'accuracy_heatmap_43.dat',
    # 'accuracy_heatmap_44.dat',
    # 'accuracy_heatmap_45.dat',
    # 'accuracy_heatmap_46.dat',
    # 'accuracy_heatmap_47.dat',
    # 'accuracy_heatmap_48.dat',
    # 'accuracy_heatmap_49.dat',
    # 'accuracy_heatmap_50.dat',
    # 'accuracy_heatmap_51.dat',
    # 'accuracy_heatmap_52.dat',
    # 'accuracy_heatmap_53.dat',
    # 'accuracy_heatmap_54.dat',
    # 'accuracy_heatmap_55.dat',
    # 'accuracy_heatmap_56.dat',
    # 'accuracy_heatmap_all.dat',
    # 'all_ap_num_1_3m.dat',
    # 'all_ap_num_2_3m.dat',
    # 'all_ap_num_3_3m.dat',
    # 'all_ap_num_4_3m.dat',
    # 'all_ap_num_5_3m.dat',
    # 'all_ap_num_6_3m.dat',
    # 'all_ap_num_7_3m.dat',
    # 'all_ap_num_8_3m.dat'
    # 'all_ap_num_1.dat',
    # 'all_ap_num_2.dat',
    # 'all_ap_num_3.dat',
    # 'all_ap_num_4.dat',
    # 'all_ap_num_5.dat',
    # 'all_ap_num_6.dat',
    # 'all_ap_num_7.dat',
    # 'all_ap_num_8.dat'
    # 'vivo_1m_4pm_0412.dat',
    # 'vivo_1m_11am_0412.dat',
    # 'vivo_2m_4pm_0412.dat',
    # 'vivo_2m_11am_0412.dat',
    # 'vivo_3m-front_4pm_0412.dat',
    # 'vivo_3m-front_11am_0412.dat',
    # 'vivo_3m-back_4pm_0412.dat',
    # 'vivo_3m-back_11am_0412.dat',
    # "8e8ab09cab814717_0503.dat",
    # "06261fb509d8ff54_0503.dat",
    # "amy_0629.dat",
    # "edmund_new.dat",
    # "guanyao.dat",
    # "jiajie_0424.dat",
    # "jiajie_0426.dat",
    # "jiajie_lohaspark.dat",
    # "jierun.dat",
    # "robert_0424.dat",
    # "steve.dat",
    # "urop.dat",
    # "willis.dat",
]
root_folder = os.getcwd()
folders = []
sub_paths = []
for sub_path in os.listdir(root_folder):
    path = os.path.join(root_folder, sub_path)
    if os.path.isdir(path):
        folders.append(path)
        sub_paths.append(sub_path)
idxs = [int(item) for item in input(f"{list(enumerate(sub_paths))}\nplease choose the folder index you want to test: ").split(" ")]
# idxs = [28]
# folder = folders[idxs[0]]
# contamination = 0.019
# check_update(folder, 'robert_0424', parts, rep_size, negative_ratio, offset, equal_flag, 12, ratios=[0.6, 0.7, 0.8, 0.9, 1])

# file_to_write = "results-6-10.txt"


# file_to_write = f"results-rep-size-num-modify.txt"
# file_to_write = f"results-prune.txt"
# file_to_write = f"results-contamination.txt"
# file_to_write = f"offset-{offset}.txt"
# file_to_write = "results-svdd.txt"
# file_to_write = "results-iforest.txt"
# file_to_write = "results-fb.txt"
# file_to_write = "results-lof.txt"
# models = ['svdd', 'lof', 'fb', 'iforest']
# for current_model in models:
    # file_to_write = f"results-{current_model}-prune-train.txt"

file_to_write = f"results-single-home-ap_num.txt"
with open(file_to_write, 'w') as f_out:
    # for contamination in [0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 0.013, 0.015, 0.017, 0.019, 0.021, 0.023, 0.025, 0.027, 0.029]:
    # for contamination in [0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 0.013, 0.015, 0.017, 0.019,
    #                       0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19,
    #                       0.2, 0.3, 0.4, 0.5]:
    contamination = 0.013
    for idx in idxs:
        folder = folders[idx]
        sub_path = sub_paths[idx]
        items = sub_path.split("-")
        rep_size = int(items[2])
        parts = int(items[4]) - int(items[3]) + 1

        ratios = []
        if int(items[3]) != 1:
            ratios = [0]
            for ite in range(int(items[3]), int(items[4]) + 1):
                ratios.append(ite/int(items[4]))
            equal_flag = False
        print(ratios)

        if os.path.isfile(os.path.join(folder, f"log_update_{contamination}.txt")):
            os.remove(os.path.join(folder, f"log_update_{contamination}.txt"))

        classification_report_test = []
        classification_report_all = []
        scores_test = []
        scores_all = []
        labels_test = []
        labels_all = []
        # After temperature scaling
        # classification_report_test_ts = []
        # classification_report_all_ts = []

        items = folder.split('-')
        if items[-1] == "update" or items[-1] == "train":
            prune_num = items[-2]
            if items[-1] == "update":
                train = 0
            else:
                train = 1
        else:
            prune_num = 0
            train = -1

        if items[-1] == "modify":
            num_modify = items[-2]
        else:
            num_modify = 10

        # for n_bins in range(5,20):
        n_bins = 10
        for exp_file in exp_files:
            classification_report_test.clear()
            classification_report_all.clear()
            scores_test.clear()
            scores_all.clear()
            labels_test.clear()
            labels_all.clear()
            file_id = exp_file.replace(".dat", "")
            print(f"##########{file_id}")
            if int(items[3]) != 1:
                check_update(folder, file_id, parts, rep_size, negative_ratio, offset, equal_flag, n_bins, ratios=ratios)
            else:
                check_update(folder, file_id, parts, rep_size, negative_ratio, offset, equal_flag, n_bins)
            # plot_result(classification_report_test, tested=True)
            # plot_result(classification_report_all, tested=False)
            dict_report = {"rep_size":rep_size, "num_modify":num_modify, "contamination":contamination, "prune_num":prune_num, "train":train, "n_bins":n_bins,
                         "id":file_id, "report_test":classification_report_test, "report_all":classification_report_all, "scores_test": scores_test, "scores_all": scores_all,
                           "label_test": labels_test, "label_all": labels_all}
            dict_json = json.dumps(dict_report)
            f_out.write(f"{dict_json} \n")

'''
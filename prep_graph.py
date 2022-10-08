from utils import * 
from config import prep_graph_config
import time

# First step: load file and get observation points
def load_file(filename, observation_ids, observation_sign, file_type, prune, th):
    max_id = 0
    if observation_ids:
        max_id = max(max_id, int(observation_ids[-1][0][1:]))

    with open(filename, "r") as f_in:
        while True:
            line = f_in.readline().rstrip(" \n")
            if not line:
                break
            if file_type == "path":
                if line.startswith("Start:"):
                    device_id = os.path.basename(filename)[:2]
                    breakpoints = [[float(coor) for coor in item.split(
                        ",")] for item in f_in.readline().rstrip(" \n").split(" ")]
                    timestamps = [
                        int(ts) for ts in f_in.readline().rstrip(" \n").split(" ")]
                    continue
                timestamp, rssi_pairs = line.split(" ", 1)
                ground_truth = interpolate_point(
                    int(timestamp), timestamps, breakpoints)
            elif file_type == "db":
                device_id = os.path.basename(filename)
                if len(line.split(" ", 1))>1:
                    coor, rssi_pairs = line.split(" ", 1)
                    ground_truth = [float(item) for item in coor.split(",")]
            elif file_type == "new":
                wifi_json = json.loads(line)
                timestamp = wifi_json['sysTimeMs']
                rssi_pairs = ''
                for item in wifi_json['data']:
                    rssi_pairs += str(item['bssid'].replace(':','')) + ',' + str(item['rssi']) + ' '
                rssi_pairs = rssi_pairs.strip(' ')
                ground_truth = [None,None]
                device_id = None
            rssi_dict = {}
            for rssi_pair in rssi_pairs.split(" "):
                mac = rssi_pair.split(",")[0][3:]
                rssi = rssi_pair.split(",")[1]
                # remove virtual mac
                #if prune and is_virtual_mac(mac): 
                #    continue
                if float(rssi) >= th:
                    rssi_dict[mac] = float(rssi)
            if rssi_dict:
                observation_ids.append(
                    ["{}{}".format(observation_sign, max_id+1), device_id, ground_truth, rssi_dict])
                max_id += 1

    print("{} loaded".format(filename))
    return observation_ids

# Second step: generate ap nodes from given observation points
def generate_ap_ids(ap_file, ap_sign, observation_ids):
    max_id = 0
    if os.path.isfile(ap_file):
        ap_ids = file_to_series(ap_file)
        max_id = max(max_id, int(ap_ids[-1][0][1:]))
    else:
        ap_ids = []

    for observation in observation_ids:
        for mac in observation[3].keys():
            ap_flag = False
            for ap in ap_ids:
                if ap[1] == mac:
                    ap_flag = True
                    ap[3].append([observation[3][mac]]+observation[2])
                    break
            if not ap_flag:
                ap_ids.append(["{}{}".format(ap_sign, max_id+1), mac,
                               None, [[observation[3][mac]]+observation[2]]])
                max_id += 1
    if ap_ids:
        series_to_file(ap_ids, ap_file)
    else:
        print("No AP ids!")
    return ap_ids

# Third step: use observation nodes and ap nodes to generate edgelist for further use
def generate_graph_file(observation_file, ap_file, offset, output_file):    
    observation_ids = file_to_series(observation_file)
    ap_ids = file_to_series(ap_file)
    ap_dict = {}
    for ap_item in ap_ids:
        ap_dict[ap_item[1]] = ap_item[0]

    with open(output_file, "w") as f_out:
        for observation in observation_ids:
            ob_id = observation[0]
            for mac in observation[3].keys():
                f_out.write("{} {} {}\n".format(ob_id, ap_dict[mac], rssi2weight(
                    offset, observation[3][mac])))
    print("edgelist generated.")

if __name__ == "__main__":
    # load settings
    offset = prep_graph_config["offset"]
    ap_sign = prep_graph_config["ap_sign"]
    observation_sign = prep_graph_config["obs_sign"]
    th = prep_graph_config["threshold"]
    input_dir_name = prep_graph_config["input_folder_name"]
    output_dir_name = prep_graph_config["output_folder_name"]
    target_dir_name = prep_graph_config["output_dir_name"]
        
    print("processing...")
    info_dict = {}
    root_dir = os.getcwd()
    input_dir = os.path.join(root_dir, input_dir_name)
    output_dir = os.path.join(root_dir, output_dir_name)
    # reset the output directory
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)

    for raw_data_dir in os.listdir(input_dir):
        item = os.path.join(input_dir, raw_data_dir)

        if os.path.isdir(item):

            output_dir_dataset = os.path.join(output_dir, raw_data_dir)
            target_dir = os.path.join(output_dir_dataset, target_dir_name)
            if not os.path.isdir(output_dir_dataset):
                os.mkdir(output_dir_dataset)
                os.mkdir(target_dir)
                os.mkdir(os.path.join(output_dir_dataset, "embedding"))
                os.mkdir(os.path.join(output_dir_dataset, "anchor list"))
                os.mkdir(os.path.join(output_dir_dataset, "spring"))
                os.mkdir(os.path.join(output_dir_dataset, "anchor map"))
                os.mkdir(os.path.join(output_dir_dataset, "incremental data"))

            observation_file = os.path.join(target_dir, "_observations_{}.pkl".format(raw_data_dir))
            ap_file = os.path.join(target_dir, "_APs_{}.pkl".format(raw_data_dir))

            pruned_observation_file = os.path.join(target_dir, "_observations_prune_{}.pkl".format(raw_data_dir))
            pruned_ap_file = os.path.join(target_dir, "_APs_prune_{}.pkl".format(raw_data_dir))

            graph_file = os.path.join(target_dir, "{}.edgelist".format(raw_data_dir))
            pruned_graph_file = os.path.join(target_dir, "prune_{}.edgelist".format(raw_data_dir))

            observation_ids = []
            pruned_observation_ids = []
            for filename in os.listdir(item):
                if filename.endswith("WiFi.txt"):
                    file = os.path.join(item, filename)
                    observation_ids = load_file(file, observation_ids, observation_sign, file_type="path", prune=False, th=th)
                    pruned_observation_ids = load_file(file, pruned_observation_ids, observation_sign, file_type="path", prune=True, th=th)
                elif filename.startswith("fingerprint"):
                    file = os.path.join(item, filename)
                    observation_ids = load_file(file, observation_ids, observation_sign, file_type="db", prune=False, th=th)
                    pruned_observation_ids = load_file(file, pruned_observation_ids, observation_sign, file_type="db", prune=True, th=th)
                elif filename.startswith("wifi-"):
                    file = os.path.join(item, filename)
                    observation_ids = load_file(file, observation_ids, observation_sign, file_type="new", prune=False, th=th)
                    pruned_observation_ids = load_file(file, pruned_observation_ids, observation_sign, file_type="new", prune=True, th=th)
            
            series_to_file(observation_ids, observation_file)
            series_to_file(pruned_observation_ids, pruned_observation_file)

            ap_ids = generate_ap_ids(ap_file, ap_sign, observation_ids)
            generate_graph_file(observation_file, ap_file, offset, graph_file)

            start_time = time.time()
            pruned_ap_ids = generate_ap_ids(pruned_ap_file, ap_sign, pruned_observation_ids)
            generate_graph_file(pruned_observation_file, pruned_ap_file, offset, pruned_graph_file)
            delta_t = time.time() - start_time
            print (f"Delta t: {delta_t}")
            print (f"Time for each node: {delta_t/len(pruned_observation_ids)}")

            info_dict[raw_data_dir] = {}
            info_dict[raw_data_dir]["observation"] = len(observation_ids)
            info_dict[raw_data_dir]["ap"] = len(ap_ids)
            info_dict[raw_data_dir]["prune observation"] = len(pruned_observation_ids)
            info_dict[raw_data_dir]["prune ap"] = len(pruned_ap_ids)
    
    print("generation completed.")
    print(info_dict)

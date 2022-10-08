import json
import numpy as np

IN = 0
OUT = 1

def load_dataset(filename, mode="new", old_formatted_data={}):
    with open(filename, "r", encoding="UTF-8") as f_in:
        raw_data = json.load(f_in)

        # part 0: labels and timestamps
        label_dict = {}
        timestamp_dict = {}

        # part 1: WiFi data
        ap_dict = {} # given AP's bssid, find its index in the network embedding
        ob_dict = {} # given observation's dataid, find its index in the network embedding
        in_ap_dict = {} # store the APs that are inside

        if len(old_formatted_data) != 0:
            old_ap_dict, old_ob_dict, old_in_ap_dict = old_formatted_data["info"]["wifi-map"]
        else:
            old_ap_dict, old_ob_dict, old_in_ap_dict = {}, {}, {}

        observation_dict = {}

        # part 2: Bluetooth data
        bc_dict = {} # given beacon's bssid, find its index in the network embedding
        bt_dict = {} # given bluetooth's dataid, find its index in the network embedding
        in_bc_dict = {} # store the ibeacons that are inside
        bluetooth_dict = {}

        # part 3: Barometer data
        baro_dict = {}

        # part 4: GPS data
        gps_dict = {}

        # part 5: Cellular data

        # part 6: Raw wifi data with missing value completion
        raw_wifi_dict = {}
        ap_list = []
        in_ap_list = []

        ap_overlap = 0 # overlap between ap_dict and old_ap_dict
        for entry in raw_data["reports"]:
            entry_id = entry["dataId"]
            if "label" not in entry:
                continue
            if len(entry["data"]["nearby"]["wlan"]["list"]) == 0:
                continue
            label_dict[entry_id] = IN if entry["label"] == 1 else OUT
            timestamp_dict[entry_id] = entry["timestamp"]

            # step 1: WiFi
            if mode == "new":
                ob_id = len(ob_dict)+len(ap_dict)+len(old_ob_dict)+len(old_ap_dict) - ap_overlap
            else:
                ob_id = "u{}".format(len(ob_dict)+len(old_ob_dict)+1)
            if entry_id not in ob_dict.keys():
                ob_dict[entry_id] = ob_id
                observation_dict[ob_id] = {}
            for item in entry["data"]["nearby"]["wlan"]["list"]:
                bssid = item["bssid"]
                rssi = int(item["rssi"])

                # use ob_id and ap_id as keys for the observations and APs
                if bssid not in ap_dict.keys():
                    if mode == "new":
                        if bssid not in old_ap_dict.keys():
                            ap_dict[bssid] = len(ob_dict)+len(ap_dict)+len(old_ob_dict)+len(old_ap_dict) - ap_overlap
                        else:
                            ap_dict[bssid] = old_ap_dict[bssid]
                            ap_overlap += 1
                    else:
                        ap_dict[bssid] = "i{}".format(len(ap_dict)+len(old_ap_dict)+1)
                ap_id = ap_dict[bssid]
                observation_dict[ob_id][ap_id] = rssi
                if entry["label"] == 1:
                    in_ap_dict[bssid] = ap_id

            # step 2: Bluetooth
            # bt_id = "u{}".format(len(bt_dict)+1)
            # if entry_id not in bt_dict.keys():
            #     bt_dict[entry_id] = bt_id
            #     bluetooth_dict[bt_id] = {} 
            # for item in entry["data"]["nearby"]["beacon"]["list"]:
            #     uuid = item["uuid"]
            #     rssi = int(item["rssi"])

            #     # use bc_id and bt_id as keys for the beacon and bluetooth
            #     if uuid not in bc_dict.keys():
            #         bc_dict[uuid] = "i{}".format(len(bc_dict)+1)
            #     bc_id = bc_dict[uuid]
            #     bluetooth_dict[bt_id][bc_id] = rssi
            #     if entry["label"] == 1:
            #         in_bc_dict[uuid] = bc_id

            # step 3: Barometer
            if 'baro' in entry['data']['nearby'].keys():
                baro_dict[entry_id] = entry["data"]["nearby"]["baro"]["values"]

            # step 4: GPS
            if 'geoloc' in entry['data'].keys():
                gps_dict[entry_id] = [entry["data"]["geoloc"][key] for key in ["latitude", "longitude"]]

            # step 5: cellular

            # step 6: raw wifi
            if entry_id not in raw_wifi_dict.keys():
                raw_wifi_dict[entry_id] = {}
            for item in entry["data"]["nearby"]["wlan"]["list"]:
                bssid = item["bssid"]
                if bssid not in ap_list:
                    ap_list.append(bssid)
                rssi = int(item["rssi"])
                raw_wifi_dict[entry_id][bssid] = rssi
                if entry["label"] == 1 and bssid not in in_ap_list:
                    in_ap_list.append(bssid)

        # align length for barometer vectors
        min_length = np.min([len(item) for item in list(baro_dict.values())]) if len(baro_dict.keys()) > 0 else 0
        old_length = 0
        if len(old_formatted_data) != 0:
            if len(old_formatted_data["measurements"]["barometer"]) != 0:
                old_length =  np.min([len(item) for item in list(old_formatted_data["measurements"]["barometer"].values())])
        if old_length == 0:
            print("minimum length of barometer is 0.")
            baro_dict = {}
        else:
            if len(baro_dict) != 0:
                if min_length > old_length:
                    for entry_id in baro_dict:
                        baro_dict[entry_id] = baro_dict[entry_id][0:old_length]
                elif min_length < old_length:
                    for entry_id in baro_dict:
                        baro_dict[entry_id] = baro_dict[entry_id] + [baro_dict[entry_id][-1]] * (old_length-min_length)
            else:
                print("this data may have problems with barometer. Suggest not use barometer.")
            # for entry_id in baro_dict.keys():
            #     baro_dict[entry_id] = baro_dict[entry_id][0:min_length]

        # fill in missing GPS values
        reverse_timestamp_dict = {v:k for k,v in timestamp_dict.items()}
        final_accurate_timestamp = -1
        complete_flag = True
        if len(gps_dict.keys()) > 0:
            for timestamp in sorted(list(reverse_timestamp_dict.keys())):
                entry_id = reverse_timestamp_dict[timestamp]
                if gps_dict[entry_id] != [0.0, 0.0]:
                    final_accurate_timestamp = timestamp
                else:
                    complete_flag = False
        # case 1: all GPS are missing
        if final_accurate_timestamp == -1:
            gps_dict = {}
        # case 2: not all GPS are present (normal cases)
        elif not complete_flag:
            last_accurate_timestamp = -1
            for timestamp in sorted(list(reverse_timestamp_dict.keys())):
                entry_id = reverse_timestamp_dict[timestamp]
                if gps_dict[entry_id] != [0.0, 0.0]:
                    # some missing values between
                    if last_accurate_timestamp != -1:
                        last_entry_id = reverse_timestamp_dict[last_accurate_timestamp]
                        # find missing points
                        missing_timestamps = sorted(list([ts for ts in reverse_timestamp_dict.keys()
                                                          if last_accurate_timestamp < ts < timestamp]))
                        gps_diff = [(gps_dict[entry_id][item] - gps_dict[last_entry_id][item])/(len(missing_timestamps)+1) for item in [0,1]]
                        # interpolate missing points
                        for idx, ts in enumerate(missing_timestamps):
                            fill_entry_id = reverse_timestamp_dict[ts]
                            gps_dict[fill_entry_id] = [(gps_dict[last_entry_id][item] + gps_diff[item]* (idx+1)) for item in [0,1]]
                    last_accurate_timestamp = timestamp
                # missing at the end
                elif timestamp > final_accurate_timestamp:
                    gps_dict[entry_id] = gps_dict[reverse_timestamp_dict[final_accurate_timestamp]]

        formatted_data = {"info":{}, "measurements":{}}
        formatted_data["info"] = {"label": label_dict, "timestamp": timestamp_dict, "wifi-map": [ap_dict, ob_dict, in_ap_dict], "bluetooth-map": [bc_dict, bt_dict, in_bc_dict], "raw-wifi-map": [ap_list, in_ap_list]}
        formatted_data["measurements"] = {"wifi": observation_dict, "bluetooth": bluetooth_dict, "barometer": baro_dict, "gps": gps_dict, "cellular": {}, "raw-wifi": raw_wifi_dict}

        return formatted_data

def merge_formatted_data(old_formatted_data, formatted_data):
    def merge_dict(dict1, dict2):
        dictMerged2 = dict1.copy()
        dictMerged2.update(dict2)
        return dictMerged2
    def merge_list(list1, list2):
        return list(set(list1 + list2))
    merged_data = {"info":{}, "measurements":{}}
    merged_data["info"] = {
        "label": merge_dict(old_formatted_data["info"]["label"], formatted_data["info"]["label"]),
        "timestamp": merge_dict(old_formatted_data["info"]["timestamp"], formatted_data["info"]["timestamp"]),
        "wifi-map":[
            merge_dict(old_formatted_data["info"]["wifi-map"][idx], formatted_data["info"]["wifi-map"][idx]) for idx in range(3)
        ],
        "bluetooth-map":[
            merge_dict(old_formatted_data["info"]["bluetooth-map"][idx], formatted_data["info"]["bluetooth-map"][idx]) for idx in range(3)
        ],
        "raw-wifi-map":[
            merge_list(old_formatted_data["info"]["raw-wifi-map"][idx], formatted_data["info"]["raw-wifi-map"][idx]) for idx in range(2)
        ]
    }

    merged_data["measurements"] = {
        "wifi": merge_dict(old_formatted_data["measurements"]["wifi"], formatted_data["measurements"]["wifi"]),
        "bluetooth": merge_dict(old_formatted_data["measurements"]["bluetooth"], formatted_data["measurements"]["bluetooth"]),
        "barometer": merge_dict(old_formatted_data["measurements"]["barometer"], formatted_data["measurements"]["barometer"]),
        "gps": merge_dict(old_formatted_data["measurements"]["gps"], formatted_data["measurements"]["gps"]),
        "cellular": {},
        "raw-wifi": merge_dict(old_formatted_data["measurements"]["raw-wifi"], formatted_data["measurements"]["raw-wifi"])
    }

    return merged_data

prep_node_config = dict(
    input_folder_name = "raw",
    output_folder_name = "input",
)

prep_graph_config = dict(
    offset = 120,
    ap_sign = "i",
    obs_sign = "u",
    threshold = -90,
    input_folder_name = "input",
    output_folder_name = "emb_output",
    output_dir_name = "raw_data",
)

emb_config = dict(
    dim = 32,
    id = "lab_test",
    dir_name = "emb_output",
    lr = 0.003,
)

update_config = dict(
    parts = 1,
    rep_size = 32,
    neg_ratio = 4,
    offset = 120,
    equal_flag = True,
    n_bin = 16,
    dim = 32,
    threshold_u = 0.005,
    temperature = 0.06,
    threshold_l = 0.001,
    contamination = 0.001,
    data_name = "lab",
    output_dir_name = "result",
    emb_dir_name = "emb_output",
    log_dir_name = "update_output",
    summary_dir_name = "summary",
)
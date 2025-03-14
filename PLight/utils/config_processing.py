import os
import shutil
import json
import copy


def memo_rename(traffic_file_list):
    new_name = ""
    for traffic_file in traffic_file_list:
        if "synthetic" in traffic_file:
            sta = traffic_file.rfind("-") + 1
            print(traffic_file, int(traffic_file[sta:-4]))
            new_name = new_name + "syn" + traffic_file[sta:-4] + "_"
        elif "cross" in traffic_file:
            sta = traffic_file.find("equal_") + len("equal_")
            end = traffic_file.find(".xml")
            new_name = new_name + "uniform" + traffic_file[sta:end] + "_"
        elif "flow" in traffic_file:
            new_name = traffic_file[:-4]
    new_name = new_name[:-1]
    return new_name


def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)

    return dic_result


def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i

    return -1


def copy_conf_file(dic_path, dic_exp_conf, dic_traffic_env_conf):
    # write conf files

    path = dic_path["PATH_TO_WORK_DIRECTORY"]
    json.dump(dic_exp_conf, open(os.path.join(path, "exp.conf"), "w"),
              indent=4)
    json.dump(dic_traffic_env_conf,
              open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)


def copy_anon_file(dic_path, dic_exp_conf):
    # hard code !!!

    path = dic_path["PATH_TO_WORK_DIRECTORY"]
    # copy sumo files

    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_exp_conf["TRAFFIC_FILE"][0]),
                os.path.join(path, dic_exp_conf["TRAFFIC_FILE"][0]))
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_exp_conf["ROADNET_FILE"]),
                os.path.join(path, dic_exp_conf["ROADNET_FILE"]))


def path_check(dic_path):
        # check path
        if os.path.exists(dic_path["PATH_TO_WORK_DIRECTORY"]):
            if dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(dic_path["PATH_TO_WORK_DIRECTORY"])

        if os.path.exists(dic_path["PATH_TO_MODEL"]):
            if dic_path["PATH_TO_MODEL"] != "model/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(dic_path["PATH_TO_MODEL"])

        if os.path.exists(dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"]):
            pass
        else:
            os.makedirs(dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"])

        if os.path.exists(dic_path["PATH_TO_PRETRAIN_MODEL"]):
            pass
        else:
            os.makedirs(dic_path["PATH_TO_PRETRAIN_MODEL"])

# 更改配置文件，会在原dic_conf配置文件中对字段进行修改或者更新

import config.dic_conf as dc
import os
import time
import utils.config_processing as cp

# import progressbar


# from script import get_traffic_volume

# multi_process = True
# 全局变量，这些变量如果在全面使用global声明后，就可以对预定义的值进行修改，如果调用了parse——
TOP_K_ADJACENCY = -1
TOP_K_ADJACENCY_LANE = -1
PRETRAIN = False
EARLY_STOP = False
NEIGHBOR = False
SAVEREPLAY = False
ADJACENCY_BY_CONNECTION_OR_GEO = False
hangzhou_archive = True
ANON_PHASE_REPRE = []


def update_global_conf(parser):
    global hangzhou_archive
    hangzhou_archive = False
    global TOP_K_ADJACENCY
    TOP_K_ADJACENCY = 5  # 定义邻居数量
    global TOP_K_ADJACENCY_LANE
    TOP_K_ADJACENCY_LANE = 5  # ？不知道定义个什么东西
    global NUM_ROUNDS
    NUM_ROUNDS = 30  # 改这里才可以
    global EARLY_STOP
    EARLY_STOP = False
    global NEIGHBOR
    # TAKE CARE
    NEIGHBOR = False
    global SAVEREPLAY  # if you want to relay your simulation, set it to be True
    SAVEREPLAY = False
    global ADJACENCY_BY_CONNECTION_OR_GEO
    # TAKE CARE
    ADJACENCY_BY_CONNECTION_OR_GEO = False

    # modify:TOP_K_ADJACENCY in line 154
    global PRETRAIN
    PRETRAIN = False

    global ANON_PHASE_REPRE
    tt = parser
    if 'CoLight_Signal' in tt.mod:
        # 12dim
        ANON_PHASE_REPRE = {
            # 0: [0, 0, 0, 0, 0, 0, 0, 0],
            1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
            2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
            3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
            4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
        }
    else:
        # 12dim
        ANON_PHASE_REPRE = {
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0]
        }
    print('agent_name:%s', tt.mod)
    print('ANON_PHASE_REPRE:', ANON_PHASE_REPRE)


def get_dic_config(args):
    # main(args.memo, args.env, args.road_net, args.gui, args.volume, args.ratio, args.mod, args.cnt, args.gen)
    # Jinan_3_4

    memo = args.memo
    env = args.env
    road_net = args.road_net
    gui = args.gui
    volume = args.volume
    suffix = args.suffix
    mod = args.mod
    cnt = args.cnt
    r_all = args.all
    workers = args.workers
    onemodel = args.onemodel

    update_global_conf(args)

    NUM_COL = int(road_net.split('_')[0])
    NUM_ROW = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)

    ENVIRONMENT = ["sumo", "anon"][env]

    if r_all:
        traffic_file_list = [ENVIRONMENT + "_" + road_net + "_%d_%s" % (v, suffix) for v in range(100, 400, 100)]
    else:
        traffic_file_list = ["{0}_{1}_{2}_{3}".format(ENVIRONMENT, road_net, volume, suffix)]

    if env:
        traffic_file_list = [i + ".json" for i in traffic_file_list]
    else:
        traffic_file_list = [i + ".xml" for i in traffic_file_list]

    # process_list = []
    n_workers = workers  # len(traffic_file_list)
    # multi_process = True

    global PRETRAIN
    global NUM_ROUNDS
    global EARLY_STOP
    for traffic_file in traffic_file_list:
        dic_exp_conf_extra = {

            "RUN_COUNTS": cnt,
            "MODEL_NAME": mod,
            "TRAFFIC_FILE": [traffic_file],  # here: change to multi_traffic

            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

            "NUM_ROUNDS": NUM_ROUNDS,

            "MODEL_POOL": False,
            "NUM_BEST_MODEL": 3,

            "PRETRAIN": PRETRAIN,  #
            "PRETRAIN_MODEL_NAME": mod,
            "PRETRAIN_NUM_ROUNDS": 0,
            "PRETRAIN_NUM_GENERATORS": 15,

            "AGGREGATE": False,
            "DEBUG": False,
            "EARLY_STOP": EARLY_STOP,
        }

        dic_agent_conf_extra = {
            "EPOCHS": 100,
            "SAMPLE_SIZE": 1000,
            "MAX_MEMORY_LEN": 10000,
            "UPDATE_Q_BAR_EVERY_C_ROUND": False,
            "UPDATE_Q_BAR_FREQ": 5,
            # network

            "N_LAYER": 2,
            "TRAFFIC_FILE": traffic_file,
        }

        global TOP_K_ADJACENCY
        global TOP_K_ADJACENCY_LANE
        global NEIGHBOR
        global SAVEREPLAY
        global ADJACENCY_BY_CONNECTION_OR_GEO
        global ANON_PHASE_REPRE

        dic_traffic_env_conf_extra = {
            "USE_LANE_ADJACENCY": True,
            "ONE_MODEL": onemodel,  # 传参
            "NUM_AGENTS": num_intersections,  # 函数前面得到的
            "NUM_INTERSECTIONS": num_intersections,  # 根据参数获得的数据
            "ACTION_PATTERN": "set",
            "MEASURE_TIME": 10,
            "IF_GUI": gui,  # 传参
            "DEBUG": False,
            "TOP_K_ADJACENCY": TOP_K_ADJACENCY,  # 全局的变量，在最上面
            "ADJACENCY_BY_CONNECTION_OR_GEO": ADJACENCY_BY_CONNECTION_OR_GEO,  # 全局的变量，在最上面
            "TOP_K_ADJACENCY_LANE": TOP_K_ADJACENCY_LANE,  # 全局的变量，在最上面
            "SIMULATOR_TYPE": ENVIRONMENT,  # 根据参数获得的数据
            "BINARY_PHASE_EXPANSION": True,
            "FAST_COMPUTE": True,

            "NEIGHBOR": NEIGHBOR,  # 全局的变量，在最上面
            "MODEL_NAME": mod,  # 传参

            "SAVEREPLAY": SAVEREPLAY,  # 全局的变量，在最上面
            "NUM_ROW": NUM_ROW,  # 参数获得
            "NUM_COL": NUM_COL,  # 参数获得

            "TRAFFIC_FILE": traffic_file,  # 'anon_3_4_jinan_4.json' for循环的产物？这个不清楚诶，为什么每个循环下面这个都不同呢？，就一个，然后表示的是路网配置文件，这个应该是固定的
            "VOLUME": volume,  # 传参
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),  # 传参

            "phase_expansion": {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0],
                5: [1, 1, 0, 0, 0, 0, 0, 0],
                6: [0, 0, 1, 1, 0, 0, 0, 0],
                7: [0, 0, 0, 0, 0, 0, 1, 1],
                8: [0, 0, 0, 0, 1, 1, 0, 0]
            },

            "phase_expansion_4_lane": {
                1: [1, 1, 0, 0],
                2: [0, 0, 1, 1],
            },

            "LIST_STATE_FEATURE": [
                "cur_phase",
                # "time_this_phase",
                # "vehicle_position_img",
                # "vehicle_speed_img",
                # "vehicle_acceleration_img",
                # "vehicle_waiting_time_img",
                "lane_num_vehicle",
                # "lane_num_vehicle_been_stopped_thres01",
                # "lane_num_vehicle_been_stopped_thres1",
                # "lane_queue_length",
                # "lane_num_vehicle_left",
                # "lane_sum_duration_vehicle_left",
                # "lane_sum_waiting_time",
                # "terminal",
                # "coming_vehicle",
                # "leaving_vehicle",
                # "pressure"

                # "adjacency_matrix",
                # "lane_queue_length",
                # "connectivity",

                # adjacency_matrix_lane
            ],

            "DIC_FEATURE_DIM": dict(
                D_LANE_QUEUE_LENGTH=(4,),
                D_LANE_NUM_VEHICLE=(4,),

                D_COMING_VEHICLE=(12,),
                D_LEAVING_VEHICLE=(12,),

                D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
                D_CUR_PHASE=(1,),
                D_NEXT_PHASE=(1,),
                D_TIME_THIS_PHASE=(1,),
                D_TERMINAL=(1,),
                D_LANE_SUM_WAITING_TIME=(4,),
                D_VEHICLE_POSITION_IMG=(4, 60,),
                D_VEHICLE_SPEED_IMG=(4, 60,),
                D_VEHICLE_WAITING_TIME_IMG=(4, 60,),

                D_PRESSURE=(1,),

                D_ADJACENCY_MATRIX=(2,),

                D_ADJACENCY_MATRIX_LANE=(6,),

            ),

            "DIC_REWARD_INFO": {
                "flickering": 0,  # -5,#
                "sum_lane_queue_length": 0,
                "sum_lane_wait_time": 0,
                "sum_lane_num_vehicle_left": 0,  # -1,#
                "sum_duration_vehicle_left": 0,
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1": -0.25,
                "pressure": 0  # -0.25
            },

            "LANE_NUM": {
                "LEFT": 1,
                "RIGHT": 1,
                "STRAIGHT": 1
            },

            "PHASE": {
                "sumo": {
                    0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
                    1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
                    2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
                    3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
                },

                # "anon": {
                #     # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                #     1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],# 'WSES',
                #     2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],# 'NSSS',
                #     3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],# 'WLEL',
                #     4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]# 'NLSL',
                #     # 'WSWL',
                #     # 'ESEL',
                #     # 'WSES',
                #     # 'NSSS',
                #     # 'NSNL',
                #     # 'SSSL',
                # },
                "anon": ANON_PHASE_REPRE,  # 全局的变量，在最上面
                # "anon": {
                #     # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                #     1: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                #     2: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                #     3: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                #     4: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                #     # 'WSWL',
                #     # 'ESEL',
                #     # 'WSES',
                #     # 'NSSS',
                #     # 'NSNL',
                #     # 'SSSL',
                # },
            }
        }

        ## ==================== multi_phase ====================
        global hangzhou_archive  # 以下的值会对字典里面的属性进行更改
        if hangzhou_archive:
            template = 'Archive+2'
        elif volume == 'jinan':
            template = "Jinan"
        elif volume == 'hangzhou':
            template = 'Hangzhou'
        elif volume == 'newyork':
            template = 'NewYork'
        elif volume == 'chacha':
            template = 'Chacha'
        elif volume == 'mydata':
            template = 'mydata'
        elif volume == 'dynamic_attention':
            template = 'dynamic_attention'
        elif volume == 'syn':
            template = 'Synthesis'
        elif dic_traffic_env_conf_extra["LANE_NUM"] == dc._LS:
            template = "template_ls"
        elif dic_traffic_env_conf_extra["LANE_NUM"] == dc._S:
            template = "template_s"
        elif dic_traffic_env_conf_extra["LANE_NUM"] == dc._LSR:
            template = "template_lsr"
        else:
            raise ValueError

        if dic_traffic_env_conf_extra['NEIGHBOR']:
            list_feature = dic_traffic_env_conf_extra["LIST_STATE_FEATURE"].copy()
            for feature in list_feature:
                for i in range(4):
                    dic_traffic_env_conf_extra["LIST_STATE_FEATURE"].append(feature + "_" + str(i))

        if mod in ['CoLight', 'GCN', 'SimpleDQNOne']:
            dic_traffic_env_conf_extra["NUM_AGENTS"] = 1
            dic_traffic_env_conf_extra['ONE_MODEL'] = False
            if "adjacency_matrix" not in dic_traffic_env_conf_extra['LIST_STATE_FEATURE'] and \
                    "adjacency_matrix_lane" not in dic_traffic_env_conf_extra['LIST_STATE_FEATURE'] and \
                    mod not in ['SimpleDQNOne']:
                dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("adjacency_matrix")
                dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("adjacency_matrix_lane")
                if dic_traffic_env_conf_extra['ADJACENCY_BY_CONNECTION_OR_GEO']:
                    TOP_K_ADJACENCY = 5
                    dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("connectivity")
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CONNECTIVITY'] = \
                        (5,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX'] = \
                        (5,)
                else:
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX'] = \
                        (dic_traffic_env_conf_extra['TOP_K_ADJACENCY'],)

                if dic_traffic_env_conf_extra['USE_LANE_ADJACENCY']:
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX_LANE'] = \
                        (dic_traffic_env_conf_extra['TOP_K_ADJACENCY_LANE'],)
        else:
            dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]

        if dic_traffic_env_conf_extra['BINARY_PHASE_EXPANSION']:
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE'] = (8,)
            if dic_traffic_env_conf_extra['NEIGHBOR']:
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)
            else:

                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)

        print(traffic_file)
        prefix_intersections = str(road_net)
        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", memo, traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S',
                                                                                            time.localtime(
                                                                                                time.time()))),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", memo, traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S',
                                                                                                       time.localtime(
                                                                                                           time.time()))),

            "PATH_TO_DATA": os.path.join("data", template, prefix_intersections),
            "PATH_TO_PRETRAIN_MODEL": os.path.join("model", "initial", traffic_file),
            "PATH_TO_PRETRAIN_WORK_DIRECTORY": os.path.join("records", "initial", traffic_file),
            "PATH_TO_ERROR": os.path.join("errors", memo)
        }

        deploy_dic_exp_conf = cp.merge(dc.DIC_EXP_CONF, dic_exp_conf_extra)
        deploy_dic_agent_conf = cp.merge(getattr(dc, "DIC_{0}_AGENT_CONF".format(mod.upper())),
                                      dic_agent_conf_extra)
        deploy_dic_traffic_env_conf = cp.merge(dc.dic_traffic_env_conf, dic_traffic_env_conf_extra)

        # TODO add agent_conf for different agents
        # deploy_dic_agent_conf_all = [deploy_dic_agent_conf for i in range(deploy_dic_traffic_env_conf["NUM_AGENTS"])]

    # 从这一行开始之后结束循环
    deploy_dic_path = cp.merge(dc.DIC_PATH, dic_path_extra)
    cp.path_check(deploy_dic_path)

    cp.copy_conf_file(deploy_dic_path, deploy_dic_exp_conf, deploy_dic_traffic_env_conf)
    cp.copy_anon_file(deploy_dic_path, deploy_dic_exp_conf)

    return deploy_dic_exp_conf, deploy_dic_agent_conf, deploy_dic_traffic_env_conf, deploy_dic_path

    # run(dic_exp_conf=deploy_dic_exp_conf,
    #                 dic_agent_conf=deploy_dic_agent_conf,
    #                 dic_traffic_env_conf=deploy_dic_traffic_env_conf,
    #                 dic_path=deploy_dic_path)

    # return memo





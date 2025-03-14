# 特征处理
import numpy as np
from tensorflow.keras.utils import to_categorical


def compute_len_feature(dic_traffic_env_conf, num_lanes):
    from functools import reduce
    len_feature = tuple()
    for feature_name in dic_traffic_env_conf["LIST_STATE_FEATURE"]:
        if "adjacency" in feature_name:
            continue
        elif "phase" in feature_name:
            len_feature += dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_" + feature_name.upper()]
        elif feature_name == "lane_num_vehicle":
            len_feature += (
                dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_" + feature_name.upper()][0] * num_lanes,)
    return sum(len_feature)

def adjacency_index2matrix(adjacency_index, num_agents):
    # adjacency_index(the nearest K neighbors):[1,2,3]
    """
    if in 1*6 aterial and
        - the 0th intersection,then the adjacency_index should be [0,1,2,3]
        - the 1st intersection, then adj [0,3,2,1]->[1,0,2,3]
        - the 2nd intersection, then adj [2,0,1,3]

    """
    # [batch,agents,neighbors]
    adjacency_index_new = np.sort(adjacency_index, axis=-1)
    l = to_categorical(adjacency_index_new, num_classes=num_agents)
    return l

def get_feature(state,num_agents,num_neighbors, dic_traffic_env_conf):
    # state:[batch,agent,features and adj]
    # return:act:[batch,agent],att:[batch,layers,agent,head,neighbors]
    batch_size = len(state)
    total_features, total_adjs = list(), list()
    for i in range(batch_size):
        feature = []
        adj = []
        for j in range(num_agents):
            observation = []
            for feature_name in dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if 'adjacency' in feature_name:
                    continue
                if feature_name == "cur_phase":
                    if len(state[i][j][feature_name]) == 1:
                        # choose_action
                        observation.extend(
                            dic_traffic_env_conf['PHASE'][dic_traffic_env_conf['SIMULATOR_TYPE']]
                            [state[i][j][feature_name][0]])
                    else:
                        observation.extend(state[i][j][feature_name])
                elif feature_name == "lane_num_vehicle":
                    observation.extend(state[i][j][feature_name])
            feature.append(observation)
            adj.append(state[i][j]['adjacency_matrix'])
        total_features.append(feature)
        total_adjs.append(adj)
    # feature:[agents,feature]
    total_features = np.reshape(np.array(total_features), [batch_size, num_agents, -1])
    total_adjs = adjacency_index2matrix(np.array(total_adjs),num_agents)
    return total_features, total_adjs


def get_transitions(batch_size, batch, num_intersections, num_neighbors, len_feature, dic_traffic_env_conf):
    _obs = np.ones((batch_size, num_intersections, len_feature))
    _next_obs = np.ones((batch_size, num_intersections, len_feature))
    _adjs = np.ones((batch_size, num_intersections,num_neighbors, num_intersections))
    _next_adjs = np.ones((batch_size, num_intersections, num_neighbors, num_intersections))
    _reward = np.zeros((batch_size, num_intersections))
    _action = np.ones((batch_size, num_intersections))  # 从经验回放中选择样本动作
    _done = np.zeros((batch_size, num_intersections, 1))
    for j in range(batch_size):
        sample = batch[j]
        _state = sample[0]
        _action[j] = sample[1]
        _reward[j] = sample[2]
        _next_state = sample[3]
        _done[j] = sample[4]
        _obs[j], _adjs[j] = get_feature([_state], num_intersections, num_neighbors, dic_traffic_env_conf)
        _next_obs[j], _next_adjs[j] = get_feature([_next_state], num_intersections, num_neighbors, dic_traffic_env_conf)

        transitions_dic = {
            'state': _obs,
            'adjs': _adjs,
            'next_state': _next_obs,
            'next_adjs': _next_adjs,
            'action': _action,
            'reward': _reward,
            'done': _done
        }

        return transitions_dic
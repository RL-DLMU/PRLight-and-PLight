import numpy as np
import os
import torch
import argparse
from buffer import ReplayBuffer
import utils.feature_processing as fp
import config.dic_conf as dc
import config.get_conf as gc
from EDQN import EDQN, Encoder
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import utils.down_sample as ds
import utils.cmp_avg_que as cque


train_reward_path = 'train_reward/init1tran_jinan3.txt'
test_reward_path = 'test_reward/init1tran_jinan3.txt'
test_travel_time_path = 'test_reward/travel_time/init1_tran3_travel_time3.txt'
test_queue_path = 'test_reward/queue/init1_tran3_queue.txt'
test_throughput_path = 'test_reward/throughput/init1_tran3_throughput.txt'
use_policy1_num_path = 'sample_number/init1_tran3_use1.txt'
use_policy2_num_path = 'sample_number/init1_tran3_use2.txt'
use_policy3_num_path = 'sample_number/init1_tran3_use3.txt'


def parse_args():

    parser = argparse.ArgumentParser()
    # The file folder to create/log in
    parser.add_argument("--memo", type=str, default='jinan_init1_tran3')
    parser.add_argument("--env", type=int, default=1)  # env=1 means you will run CityFlow
    parser.add_argument("--gui", type=bool, default=False)
    parser.add_argument("--road_net", type=str, default='3_4')  # which road net you are going to run
    parser.add_argument("--volume", type=str, default='jinan')  # which road net you are going to run
    parser.add_argument("--suffix", type=str, default="3")  # which flow data you are going to run
    parser.add_argument("--mod", type=str, default='CoLight')  # using CoLight's parameter configuration
    parser.add_argument("--cnt", type=int, default=3600)  # 3600
    parser.add_argument("-all", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--onemodel", type=bool, default=False)
    parser.add_argument("--visible_gpu", type=str, default="-1")

    return parser.parse_args()


def get_actions(obs, adjs, model_pool, agent, exp_id, plug_action):
    all_actions = []
    for expert in model_pool:
        q, _ = expert(obs, plug_action, adjs)
        q = q[0].cpu().detach().numpy()
        actions = np.expand_dims(np.argmax(q, axis=-1), axis=-1)
        all_actions.append(actions)
    tar_q, _ = agent(obs, plug_action, adjs)
    tar_q = tar_q[0].cpu().detach().numpy()
    tar_actions = np.expand_dims(np.argmax(tar_q, axis=-1), axis=-1)
    all_actions.append(tar_actions)

    action = []
    for i in range(len(exp_id)):
        action.append(all_actions[exp_id[i]][i])

    return np.array(action)


def cmp_dd(obs, action, adjs, next_obs, next_adjs, model_pool, agent, encoder, device):

    next_obs = torch.Tensor(next_obs).to(device)
    action = np.array([action])
    action = torch.Tensor(action).to(device).unsqueeze(2)
    next_adjs = torch.Tensor(next_adjs).to(device)

    pre_o_set = []
    for expert in model_pool:
        _, pre_o = expert(obs, action, adjs)
        pre_o_set.append(pre_o)
    _, tar_pre_o = agent(obs, action, adjs)
    pre_o_set.append(tar_pre_o)
    pre_feature_set = []

    for pre_o in pre_o_set:
        pre_feature = encoder(pre_o, next_adjs)
        pre_feature_set.append(pre_feature)
    real_feature = encoder(next_obs, next_adjs)
    pre_model_o = torch.cat(pre_feature_set, 0)


    pre_model_o = pre_model_o.permute(1, 0, 2)
    real_feature = real_feature.permute(1, 0, 2)

    pdist = torch.nn.PairwiseDistance(p=2)
    exp_dis = pdist(pre_model_o, real_feature)

    return exp_dis.cpu().detach().numpy()


def run(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):

    num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']  # 16
    num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], num_agents)  # 5
    num_actions = len(dic_traffic_env_conf["PHASE"][dic_traffic_env_conf['SIMULATOR_TYPE']])  # 4
    num_lanes = np.sum(np.array(list(dic_traffic_env_conf["LANE_NUM"].values())))  # 3
    len_feature = fp.compute_len_feature(dic_traffic_env_conf, num_lanes) # 20
    hidden_dim = 32
    buff = ReplayBuffer(20000)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = EDQN(len_feature,hidden_dim,num_actions, device)
    agent.load(f"model_pool/model_1")
    agent_tar = EDQN(len_feature,hidden_dim,num_actions, device)
    agent_tar.load(f"model_pool/model_1")
    optimizer = optim.Adam(agent .parameters(), lr=0.001)  # dic_agent_conf["LEARNING_RATE"]
    batch_size = dic_agent_conf["BATCH_SIZE"]  # 32

    train_inf = [batch_size, num_agents, num_neighbors, len_feature, device, optimizer]

    encoder = Encoder(len_feature,hidden_dim)
    encoder.load_state_dict(torch.load(f"model_pool/model_avg/encoder.pt"))
    encoder.eval()
    encoder.to(device)

    # 构建策略库
    model_pool_len = 2
    model_pool = []
    for i in range(model_pool_len):
        pre_model = EDQN(len_feature, hidden_dim, num_actions, device)
        pre_model.load(f"model_pool/model_{i+1}")
        pre_model.test()
        model_pool.append(pre_model)

    f = open(train_reward_path, 'w')
    fr = open(test_reward_path, 'w')
    ft = open(test_travel_time_path, 'w')
    fq = open(test_queue_path, 'w')
    fo = open(test_throughput_path, 'w')
    f1 = open(use_policy1_num_path, 'w')
    f2 = open(use_policy2_num_path, 'w')
    f3 = open(use_policy3_num_path, 'w')

    for iter in range(10):
        with tqdm(total=int(dic_exp_conf["NUM_ROUNDS"]/10), desc='Iteration %d' % iter) as pbra:
            for cnt_round in range(int(dic_exp_conf["NUM_ROUNDS"]/10)):

                path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "train_round", "round_" + str(cnt_round))

                env = dc.DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
                    path_to_log=path_to_log,
                    path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
                    dic_traffic_env_conf=dic_traffic_env_conf)
                env.reset()

                path = dic_path["PATH_TO_MODEL"]

                epsilon = 0.05

                step_num = 0
                use1 = 0
                use2 = 0
                use3 = 0
                reward_list = []
                done = False
                state = env.reset()

                exp_id = np.random.randint(0, model_pool_len+1, num_agents)
                step_five_exp_dis = np.zeros((num_agents, model_pool_len+1))
                plug_action = torch.zeros((1, num_agents, 1)).to(device)
                while not done and step_num < int(dic_exp_conf["RUN_COUNTS"] / dic_traffic_env_conf["MIN_ACTION_TIME"]):
                    obs, adjs = fp.get_feature([state], num_agents, num_neighbors, dic_traffic_env_conf)
                    step_num += 1

                    obs = torch.Tensor(obs).to(device)
                    adjs = torch.Tensor(adjs).to(device)
                    max_action = get_actions(obs, adjs, model_pool, agent, exp_id, plug_action)
                    random_action = np.reshape(np.random.randint(num_actions, size=1 * num_agents), (num_agents, 1))
                    possible_action = np.concatenate([max_action, random_action], axis=-1)
                    selection = np.random.choice([0, 1], size=num_agents, p=[1 - epsilon, epsilon])
                    action = possible_action.reshape((num_agents, 2))[np.arange(num_agents), selection]

                    next_state, reward, done, _ = env.step(action)

                    buff.add(state, action, reward, next_state, done)

                    # 计算下一组exp_id
                    next_obs, next_adjs = fp.get_feature([next_state], num_agents, num_neighbors, dic_traffic_env_conf)
                    exp_dis = cmp_dd(obs, action, adjs, next_obs, next_adjs, model_pool, agent, encoder, device)
                    ration = pow(0.8, (5-step_num))
                    step_five_exp_dis += ration * exp_dis
                    step_five_exp_dis += exp_dis
                    if step_num % 5 == 0:
                        step_five_exp_dis = -step_five_exp_dis
                        for k in range(len(exp_id)):
                            probabilities = F.softmax(torch.tensor(step_five_exp_dis[k]), dim=0)
                            probabilities = probabilities.cpu().detach().numpy()
                            exp_id[k] = np.random.choice(len(probabilities), p=probabilities)  # 求得距离最近的pre-model的索引

                        for number in exp_id:
                            if number == 0:
                                use1 += 1
                            elif number == 1:
                                use2 += 1
                            elif number == 2:
                                use3 += 1
                        
                        step_five_exp_dis = np.zeros((num_agents, model_pool_len+1))

                    state = next_state
                    reward_list.append(reward)

                    if buff.num_experiences >= buff.mini_size:
                        agent, agent_tar = update(buff, train_inf, agent, agent_tar, step_num)

                reward_list = np.array(reward_list)
                sum_reward = np.sum(reward_list,axis=0)
                reward = np.mean(sum_reward)

                pbra.set_postfix({'episode': '%d' % (dic_exp_conf["NUM_ROUNDS"]/10 * iter + cnt_round+1), 'return': '%.3f' % reward})
                pbra.update(1)

                f.write(str(reward)+'\n')
                f1.write(str(use1)+'\n')
                f2.write(str(use2)+'\n')
                f3.write(str(use3)+'\n')
                
                reward_test, t_time, a_que, p_throughput = train_test(dic_exp_conf, num_agents, num_neighbors, device, dic_traffic_env_conf, dic_path, agent)
                fr.write(str(reward_test) + '\n')
                ft.write(str(t_time) + '\n')
                fq.write(str(a_que) + '\n')
                fo.write(str(p_throughput) + '\n')
                
                if cnt_round % 5 == 0:
                    agent_tar.load_state_dict(agent.state_dict())

                if os.path.exists(path):
                    agent.save(path, cnt_round)
                else:
                    os.makedirs(path)
                agent.save(path, cnt_round)


def update(buff, train_inf, agent, agent_tar, step):
    batch_size = train_inf[0]
    num_agents = train_inf[1]
    num_neighbors = train_inf[2]
    len_feature = train_inf[3]
    device = train_inf[4]
    optimizer = train_inf[5]
    
    _obs = np.ones((batch_size, num_agents, len_feature))
    _action = np.ones((batch_size, num_agents))
    _next_obs = np.ones((batch_size, num_agents, len_feature))
    _adjs = np.ones((batch_size, num_agents, num_neighbors, num_agents))
    _next_adjs = np.ones((batch_size, num_agents, num_neighbors, num_agents))
    _reward = np.zeros((batch_size, num_agents, 1))
    batch = buff.getBatch(batch_size)
    
    for j in range(batch_size):
        sample = batch[j]
        _state = sample[0]
        _next_state = sample[3]
        _action[j] = sample[1]
        _obs[j], _adjs[j] = fp.get_feature([_state], num_agents, num_neighbors, dic_traffic_env_conf)
        _next_obs[j], _next_adjs[j] = fp.get_feature([_next_state], num_agents, num_neighbors,
                                                     dic_traffic_env_conf)
    
    actions = torch.tensor(_action, dtype=torch.float).to(device)
    actions = torch.unsqueeze(actions, dim=2)
    q_values, o_repr = agent(torch.Tensor(np.array(_obs)).to(device), actions,
                             torch.Tensor(np.array(_adjs)).to(device))
    target_q_values, _ = agent_tar(torch.Tensor(np.array(_next_obs)).to(device), actions,
                                   torch.Tensor(np.array(_next_adjs)).to(device))
    target_q_values = np.array(target_q_values.cpu().data)
    expected_q = np.ones((batch_size, num_agents, 1))
    expected_obs = _next_obs
    
    for i in range(batch_size):
        sample = batch[i]
        for j in range(num_agents):
            expected_q[i][j][0] = sample[2][j] + (1 - sample[4]) * dic_agent_conf["GAMMA"] * max(
                target_q_values[i][j])
    
    expected_q = torch.Tensor(expected_q).to(device)
    choose_action = torch.tensor(_action, dtype=torch.int64).to(device)
    q_values = q_values.gather(-1, choose_action.unsqueeze(-1))
    
    loss1 = torch.mean(F.mse_loss(expected_q, q_values))
    loss2 = (o_repr - torch.Tensor(expected_obs).to(device)).pow(2).mean()
    loss = loss1 + loss2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (step + 1) % 50 == 0:
        agent_tar.load_state_dict(agent.state_dict())
    
    return agent, agent_tar


def train_test(dic_exp_conf, num_agents, num_neighbors, device, dic_traffic_env_conf, dic_path, agent):
    path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test")
    if not os.path.exists(path_to_log):
        os.makedirs(path_to_log)

    env = dc.DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
        path_to_log=path_to_log,
        path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
        dic_traffic_env_conf=dic_traffic_env_conf)
    env.reset()

    step_num = 0
    reward_list = []
    done = False
    state = env.reset()

    plug_action = torch.zeros((1, num_agents, 1)).to(device)
    while not done and step_num < int(dic_exp_conf["RUN_COUNTS"] / dic_traffic_env_conf["MIN_ACTION_TIME"]):
        obs, adjs = fp.get_feature([state], num_agents, num_neighbors, dic_traffic_env_conf)
        step_num += 1
        q, _ = agent(torch.Tensor(obs).to(device), plug_action, torch.Tensor(adjs).to(device))
        q = q[0].cpu().detach().numpy()
        max_action = np.expand_dims(np.argmax(q, axis=-1), axis=-1)
        action = max_action.flatten()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        reward_list.append(reward)

    env.bulk_log_multi_process()
    ds.downsample_for_system(path_to_log, dic_traffic_env_conf)

    travel_time = env.average_travel_time()
    throughput = env.get_cur_throughput()
    reward_list = np.array(reward_list)
    sum_reward = np.sum(reward_list, axis=0)
    reward = np.mean(sum_reward)
    env.bulk_log_multi_process()
    ds.downsample_for_system(path_to_log, dic_traffic_env_conf)

    avg_que = cque.out_avg_que(num_agents, path_to_log)

    print(f'reward: {reward} | travel time: {travel_time} | avg queue: {avg_que} | throughput: {throughput} | ')

    return reward, travel_time, avg_que, throughput


if __name__ == "__main__":
    args = parse_args()
    dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = gc.get_dic_config(args)
    run(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path)

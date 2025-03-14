import numpy as np
import os
import torch
import argparse
from buffer import ReplayBuffer
import utils.feature_processing as fp
import config.dic_conf as dc
import config.get_conf as gc
from EDQN import EDQN
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import utils.down_sample as ds
import utils.cmp_avg_que as cque

data_flag = 'xxxx' # date
region_flag = 'jinan'
algo_flag = 'pre-model'

train_reward_path = f'train_reward/{algo_flag}_train_{region_flag}_{data_flag}.txt'

test_reward_path = f'test_reward/{algo_flag}_{region_flag}_{data_flag}.txt'
test_travel_time_path = f'test_reward/travel_time/{algo_flag}_test_travel_time_{region_flag}_{data_flag}.txt'
test_queue_path = f'test_reward/queue/{algo_flag}_test_queue_{region_flag}_{data_flag}.txt'
test_throughput_path = f'test_reward/throughput/{algo_flag}test_throughput_{region_flag}_{data_flag}.txt'


def parse_args():

    parser = argparse.ArgumentParser()
    # The file folder to create/log in
    parser.add_argument("--memo", type=str, default='jinan1')
    parser.add_argument("--env", type=int, default=1)  # env=1 means you will run CityFlow
    parser.add_argument("--gui", type=bool, default=False)
    parser.add_argument("--road_net", type=str, default='3_4')  # which road net you are going to run
    parser.add_argument("--volume", type=str, default='jinan')  # which road net you are going to run
    parser.add_argument("--suffix", type=str, default="1")  # which flow data you are going to run
    parser.add_argument("--mod", type=str, default='CoLight')  # using Parameter configuration of colightt
    parser.add_argument("--cnt", type=int, default=3600)  # 3600
    parser.add_argument("-all", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--onemodel", type=bool, default=False)
    parser.add_argument("--visible_gpu", type=str, default="-1")

    return parser.parse_args()


def run(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):

    num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
    num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], num_agents)
    num_actions = len(dic_traffic_env_conf["PHASE"][dic_traffic_env_conf['SIMULATOR_TYPE']])
    num_lanes = np.sum(np.array(list(dic_traffic_env_conf["LANE_NUM"].values())))
    len_feature = fp.compute_len_feature(dic_traffic_env_conf, num_lanes)
    hidden_dim = 32
    buff = ReplayBuffer(20000)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = EDQN(len_feature,hidden_dim,num_actions, device)
    agent_tar = EDQN(len_feature,hidden_dim,num_actions, device)
    optimizer = optim.Adam(agent .parameters(), lr=0.001)  # dic_agent_conf["LEARNING_RATE"]
    batch_size = dic_agent_conf["BATCH_SIZE"]  # 32
    
    train_inf = [batch_size, num_agents, num_neighbors, len_feature, device, optimizer]

    f = open(train_reward_path, 'w')
    fr = open(test_reward_path, 'w')
    ft = open(test_travel_time_path, 'w')
    fq = open(test_queue_path, 'w')
    fo = open(test_throughput_path, 'w')

    for iter in range(1):
        with tqdm(total=int(dic_exp_conf["NUM_ROUNDS"]/1), desc='Iteration %d' % iter) as pbra:
            for cnt_round in range(int(dic_exp_conf["NUM_ROUNDS"]/1)):

                path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
           
                env = dc.DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
                    path_to_log=path_to_log,
                    path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
                    dic_traffic_env_conf=dic_traffic_env_conf)
                env.reset()

                path = dic_path["PATH_TO_MODEL"]

                decayed_epsilon = dic_agent_conf["EPSILON"] * pow(dic_agent_conf["EPSILON_DECAY"], dic_exp_conf["NUM_ROUNDS"]/10 * iter + cnt_round)
                epsilon = max(decayed_epsilon, dic_agent_conf["MIN_EPSILON"])

                step_num = 0
                reward_list = []
                done = False
                state = env.reset()

                plug_action =  torch.zeros((1, num_agents, 1)).to(device)
                while not done and step_num < int(dic_exp_conf["RUN_COUNTS"] / dic_traffic_env_conf["MIN_ACTION_TIME"]):
                    obs, adjs = fp.get_feature([state], num_agents, num_neighbors, dic_traffic_env_conf)
                    step_num += 1

                    q, _ = agent(torch.Tensor(obs).to(device), plug_action, torch.Tensor(adjs).to(device))
                    q = q[0].cpu().detach().numpy()

                    max_action = np.expand_dims(np.argmax(q, axis=-1), axis=-1)
                    random_action = np.reshape(np.random.randint(num_actions, size=1 * num_agents), (num_agents, 1))
                    possible_action = np.concatenate([max_action, random_action], axis=-1)
                    selection = np.random.choice(
                        [0, 1],
                        size=num_agents,
                        p=[1 - epsilon, epsilon])
                    action = possible_action.reshape((num_agents, 2))[np.arange(num_agents), selection]
                    
                    next_state, reward, done, _ = env.step(action)

                    buff.add(state, action, reward, next_state, done)

                    state = next_state

                    reward_list.append(reward)

                    if buff.num_experiences >= buff.mini_size:
                        agent, agent_tar = update(buff, train_inf, agent, agent_tar, step_num)
                        
                reward_list = np.array(reward_list)
                sum_reward = np.sum(reward_list,axis=0)
                reward = np.mean(sum_reward)
         
                pbra.set_postfix({'episode': '%d' % (dic_exp_conf["NUM_ROUNDS"]/10 * iter + cnt_round+1), 'return': '%.3f' % reward})
                pbra.update(1)

                # 保存reward
                f.write(str(reward)+'\n')
                
                reward_test, t_time, a_que, p_throughput = train_test(dic_exp_conf, num_agents, num_neighbors, device, dic_traffic_env_conf, dic_path, agent)
                fr.write(str(reward_test) + '\n')
                ft.write(str(t_time) + '\n')
                fq.write(str(a_que) + '\n')
                fo.write(str(p_throughput) + '\n')
                
                if os.path.exists(path):
                    agent.save(path, cnt_round)
                else:
                    os.makedirs(path)


def update(buff, train_inf, agent, agent_tar, step):
    
    batch_size = train_inf[0]
    num_agents = train_inf[1]
    num_neighbors = train_inf[2]
    len_feature = train_inf[3]
    device = train_inf[4]
    optimizer = train_inf[5]
    
    _obs = np.ones((batch_size, num_agents, len_feature))
    _action = np.ones((batch_size, num_agents))  # 从经验回放中选择样本动作
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

import pickle
import numpy as np
import pandas as pd
import os
datas=[]
logs=[]

def out_quefile(num_agent, path_to_log):

    time_len=[]
    for i in range(num_agent):
        f = open(os.path.join(path_to_log, "inter_{0}.pkl".format(i)),'rb')
        data=pickle.load(f,encoding='UTF-8')
        time_len.append(len(data))
        f.close()
        datas.append(data)

    for k in range(360):
        mean=[]
        for i in range(num_agent):
            if time_len[i]==360:
                a=k
            else: a=k*10
            mean.append(np.mean(datas[i][a]["state"]['lane_num_vehicle_been_stopped_thres1']))

        if mean!=[]:
            log={"步数":k,
                 "平均队列长度":np.mean(mean)}
            logs.append(log)

    dd = pd.DataFrame(logs)
    dd.to_csv(f"policy_{id+1}_que.csv")


# policy_len = 4
# for m in range(policy_len):
#     out_quefile(m)
out_quefile(5)
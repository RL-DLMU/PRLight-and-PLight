import pickle
import numpy as np
import pandas as pd
import os


def out_avg_que(num_agent, path_to_log):
    
    inter_que_num = []
    for i in range(num_agent):
        f = open(os.path.join(path_to_log, "inter_{0}.pkl".format(i)),'rb')
        bb=pickle.load(f,encoding='UTF-8')
        f.close()
        tt=[]
        dd=[]

        for j in range(len(bb)):
            dd.append(np.sum(bb[j]["state"]['lane_num_vehicle_been_stopped_thres1']))

        que_num=np.mean(dd)
        inter_que_num.append(que_num)

    total_sum = sum(inter_que_num)
    count = len(inter_que_num)
    average = total_sum / count
    
    return average


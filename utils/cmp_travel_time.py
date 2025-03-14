import csv
import numpy as np
import pandas as pd
import os


def get_travel_time(num_agent, path_to_log):
    flow_ID={}
    count=0
    for i in range(num_agent):
        file_name = os.path.join(path_to_log, "vehicle_inter_{0}.csv".format(i))

        with open(file_name)as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                ID=row[0]
                enter_time=row[1]
                leave_time = row[2]
                if leave_time=='nan':
                    count+=1
                    leave_time=3600
                if ID =='':
                    continue
                if flow_ID.get(ID):

                    flow_ID[ID].append(float(enter_time))
                    flow_ID[ID].append(float(leave_time))
                else:
                    flow_ID[ID]=[float(enter_time),float(leave_time)]

    car_num=len(flow_ID)
    travel={}
    sum_travel=0
    for k in flow_ID.keys():
        real_enter=min(flow_ID[k])
        real_leave=max(flow_ID[k])
        travel[k]=[real_enter]
        travel[k].append(real_leave)
        sum_travel+=real_leave-real_enter

    # print(flow_ID)
    # print(travel)
    # print("car_num",car_num)
    # print("arrive_num",car_num-count)
    real_num = car_num
    t_time = sum_travel/real_num
    arrive_num = car_num - count
    # print("avg travel_time", t_time)

    return arrive_num


# def get_travel_time(num_agent, path_to_log):
#
#     flow_ID = {}
#     for i in range(num_agent):
#         file_name = os.path.join(path_to_log, "vehicle_inter_{0}.csv".format(i))
#
#         with open(file_name) as f:
#             f_csv = csv.reader(f)
#             for row in f_csv:
#                 ID = row[0]
#                 enter_time = row[1]
#                 leave_time = row[2]
#                 if leave_time == 'nan':
#                     continue
#                 if ID == '':
#                     continue
#                 if flow_ID.get(ID):
#                     flow_ID[ID].append(float(enter_time))
#                     flow_ID[ID].append(float(leave_time))
#                 else:
#                     flow_ID[ID] = [float(enter_time), float(leave_time)]
#
#     car_num = len(flow_ID)
#     travel = {}
#     sum_travel = 0
#     for k in flow_ID.keys():
#         real_enter = min(flow_ID[k])
#         real_leave = max(flow_ID[k])
#         travel[k] = [real_enter]
#         travel[k].append(real_leave)
#         sum_travel += real_leave - real_enter
#
#     # print(flow_ID)
#     # print(travel)
#     # print("car_num",car_num)
#     # print("arrive_num",car_num-count)
#
#     t_time = sum_travel / car_num
#     # print("avg travel_time", t_time)
#
#     return t_time, car_num
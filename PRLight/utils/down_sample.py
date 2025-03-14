# log记录，环境采样日志保存，可有可无
import pickle  # 用于保存，将Python对象存储到磁盘上的文件中，并在需要时从文件中重新加载对象
import os
import traceback  # 用于获取和处理异常的跟踪信息，当程序发生异常时，traceback模块提供了一些有用的函数来获取异常的调用堆栈信息，以及生成和格式化异常的跟踪信息


def downsample(path_to_log, i):
    path_to_pkl = os.path.join(path_to_log, "inter_{0}.pkl".format(i))
    with open(path_to_pkl, "rb") as f_logging_data:
        try:
            logging_data = pickle.load(f_logging_data)
            subset_data = logging_data[::10]
            # print(subset_data)
            f_logging_data.close()
            os.remove(path_to_pkl)
            with open(path_to_pkl, "wb") as f_subset:
                try:
                    pickle.dump(subset_data, f_subset)
                except Exception as e:
                    print("----------------------------")
                    print("Error occurs when WRITING pickles when down sampling for inter {0}".format(i))
                    print('traceback.format_exc():\n%s' % traceback.format_exc())
                    print("----------------------------")
        except Exception as e:
            # print("CANNOT READ %s"%path_to_pkl)
            print("----------------------------")
            print("Error occurs when READING pickles when down sampling for inter {0}, {1}".format(i, f_logging_data))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            print("----------------------------")


def downsample_for_system(path_to_log, dic_traffic_env_conf):
    for i in range(dic_traffic_env_conf['NUM_INTERSECTIONS']):
        downsample(path_to_log, i)

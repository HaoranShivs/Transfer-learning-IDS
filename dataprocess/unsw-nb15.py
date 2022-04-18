import os
import numpy as np

from torch.utils.data import DataLoader, Dataset

class UNSW_NB15_BASE(Dataset):
    def __init__(self, root_dir, mode='Train'):
        assert mode == 'Train' or mode == 'Test'
        # self.mode = mode
        self.data_array = None
        self.to_num_column_idx = [0,1,2,3,4,5,13,47]
        self.columns_name = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 
                            'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 
                            'res_bdy_len', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat', 
                            'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
                            'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label']
        self.discrete_column = [0,1,2,3,4,5,13,35,38,41,47,48]
        self.to_num_column_dic = None
        self.data_num = 0
        self.raw_data_dir_list = [root_dir + '/' + i for i in os.listdir(root_dir)]
        self.processed_train_data_dir = root_dir + '/unsw_nb_trainset.csv'
        self.processed_test_data_dir = root_dir + '/unsw_nb_testset.csv'

    def load_raw_data(self):
        def del_unicode_start(x):
            if '\xef\xbb\xbf' in x:
                return x[3:]
            else:
                return x
        def null_string(x):
            if x == '':
                return '0'
            else:
                return x
        def garbled(x):
            if x == '0x000b':
                return '0'
            elif x == '0x000c':
                return '0'
            else:
                return x
        self.data_array = np.loadtxt(self.raw_data_dir_list[0], dtype=np.string_, delimiter=',', encoding='Latin-1', converters={0:del_unicode_start, 1:garbled, 47:null_string})
        # for i in range(1, len(self.raw_data_dir_list)):
        #     self.data_array = np.vstack((self.data_array, np.loadtxt(self.raw_data_dir_list[i], dtype=np.string_, delimiter=',', encoding='Latin-1', converters={0:del_unicode_start, 1:garbled, 47:null_string})))
        self.data_num = self.data_array.shape[0]
        print('### load raw data over ###')

    def unsw_nb_numerical(self):
        '''column [1,2,3,11,13,14,20,21,41] are discrete, need to separated
           columns need to be converted to num :[1,2,3,41]
        '''

        def string_to_num(data_array, column_index):
            unique_protocol, times = np.unique(data_array[:, column_index], axis=0, return_counts=True)
            times_sort_idx = np.argsort(times, axis=0) # descending
            times_sort_idx = times_sort_idx[::-1] # indescending
            unique_protocol = unique_protocol[times_sort_idx] # make unique_protocol indescending on times
            temp = data_array[:, column_index]
            protocol_key, protocol_value = [], [] # [keys], [values]
            for i in range(unique_protocol.shape[0]):
                temp[temp==unique_protocol[i]] = i
                protocol_key.append(unique_protocol[i])
                protocol_value.append(i)
            return protocol_key, protocol_value

        to_num_column_dic = []
        for i in self.to_num_column_idx:
            key, value = string_to_num(self.data_array, i)
            to_num_column_dic.append(dict(zip(key,value)))
            print('### ', key, value, ' ###')

        self.to_num_column_dic = dict(zip(self.to_num_column_idx, to_num_column_dic))

        self.data_array = self.data_array.astype(np.float32)

    
    def normalize(self):
        column_idx = [i for i in range(42)]
        continuous_column_idx = [i for i in column_idx if i not in self.discrete_column]

        target_part = self.data_array[:,continuous_column_idx]
        column_max = np.max(target_part, axis=0, keepdims=True)
        column_min = np.min(target_part, axis=0, keepdims=True)

        _range = column_max - column_min
        __range = np.where(_range == 0, 1e-5, _range)

        self.data_array[:,continuous_column_idx] = (self.data_array[:,continuous_column_idx] - column_min) / (__range)
        self.data_array[:,continuous_column_idx] = np.where(_range == 0, 0, self.data_array[:,continuous_column_idx])

    def disorder(self):
        np.random.shuffle(self.data_array)

    def data_process(self):
        self.unsw_nb_numerical()
        self.normalize()
        self.disorder()

    def __getitem__(self, index):
        return self.data_array[index,:47], self.data_array[index, 47], self.data_array[index, 48]


    def __len__(self):
        return self.data_array.shape[0]


    def save_data(self):
        num = int(self.data_num * 0.6)
        np.savetxt(self.processed_train_data_dir, self.data_array[:num,:], delimiter=',')
        np.savetxt(self.processed_test_data_dir, self.data_array[num:,:], delimiter=',')
    
if __name__ == '__main__':
    dataset = UNSW_NB15_BASE('E:/DataSets/UNSW-NB15 - CSV Files/dataset')
    dataset.load_raw_data()
    dataset.data_process()
    # dataset.save_data()
    # for i in range(5):
    #     print(dataset[i])
    # print(dataset[5][0].shape)
    # print(dataset.__len__())
    # print(dataset.to_num_column_dic)


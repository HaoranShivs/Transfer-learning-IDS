import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset


class CIC_IDS_2107_Base(Dataset):
    def __init__(self, root_dir, mode='Train'):
        assert mode == 'Train' or mode == 'Test'
        self.mode = mode
        self.data_array = None
        self.columns_name = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 
                             'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 
                             'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 
                             'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 
                             'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 
                             'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 
                             'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 
                             'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 
                             'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 
                             'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 
                             'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 
                             'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']
        self.to_num_column_idx = [1, 41]
        self.discrete_column = [1, 41]
        self.data_num = 0
        self.to_num_column_dic = None
        self.label_category = ['BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye', 'FTP-Patator', 'SSH-Patator', 'DoS slowloris', 'DoS Slowhttptest', 'Bot', 'Web Attack Brute Force', 'Web Attack XSS', 'Infiltration', 'Web Attack Sql Injection', 'Heartbleed']
        self.label_category = ['BENIGN', 'Dos', 'PortScan', 'DDoS', 'Brute Force', 'Bot', 'Web Attack', 'Infiltration', 'Heartbleed']
        self.label_category_index = {'NORMAL':[2], 'Dos':[3, 4, 5, 6], 'DOS':[0, 1, 7, 9, 10, 13], 'U2R':[12, 16, 17, 21], 'R2L':[8, 11, 14, 15, 18, 19, 20, 22]}
        self.raw_data_dir_list = [root_dir + '/' + i for i in os.listdir(root_dir)]
        self.sort_data_dir = root_dir + '/sort_data/'
        self.processed_train_data_dir = root_dir + '/cic_ids_2017_trainset.csv'
        self.processed_test_data_dir = root_dir + '/cic_ids_2017_testset.csv'


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


    def cic_ids2017_numerical(self):

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
        self.data_array = self.data_array.astype(np.float32)

        self.to_num_column_dic = dict(zip(self.to_num_column_idx, to_num_column_dic))


    def disorder(self):
        np.random.shuffle(self.data_array)


    # def __if_in(self, array, list):
        array = array[:, -1]
        temp = array == list[0]
        for i in range(1, len(list)):
            temp = temp + (array == list[i])
        return temp


    def get_category_data(self, category):
        indices = self.label_category[category]
        select = self.__if_in(self.data_array, indices)

        self.data_array = self.data_array[select]
        

    def Uniform_data_by_label_type(self):
        label = self.label_category.keys()
        num = int(self.data_num / len(label))
        print('totol data_num:%d, every_kind_num:%d'%(self.data_num, num))
        array_tuple = ()
        for i in label:
            indices = self.label_category[i]
            select = self.__if_in(self.data_array, indices)
            data = self.data_array[select]

            if data.shape[0] > num:
                data_array = data[:num]
            else:
                lack_num = num - data.shape[0]
                remainder = lack_num % data.shape[0]
                cycle = (lack_num - remainder) / data.shape[0]
                if cycle >= 1:
                    data_array = np.vstack((data, data))
                    for i in range(1, int(cycle)):
                        data_array = np.vstack((data_array, data))
                    data_array = np.vstack((data_array, data[:remainder]))
                else:
                    data_array = np.vstack((data, data[:remainder]))
                
            array_tuple = array_tuple + (data_array,)
        self.data_array = np.vstack(array_tuple)


    def data_process(self):
        self.cic_ids2017_numerical()
        self.normalize()
        self.disorder()
        # self.Uniform_data_by_label_type()


    def data_sort(self):
        df = pd.read_csv(self.raw_data_dir_list[0], header=0, )
        for i in range(1, len(self.raw_data_dir_list)):
            df_tem = pd.read_csv(self.raw_data_dir_list[i], header=0)
            df = df.append(df_tem)
        print('load csv over')
        df = df.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
        df = df.drop(columns=[' Fwd Header Length.1'])

        df[' Label'].replace(['DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest'], 'Dos', inplace=True)
        df[' Label'].replace(
            ['Web Attack � Brute Force', 'Web Attack � XSS', 'Web Attack � Sql Injection'], 'Web Attack', inplace=True)
        df[' Label'].replace(['FTP-Patator', 'SSH-Patator'], 'Brute Force', inplace=True)

        temp = df[' Label'].value_counts(normalize=True)
        if not os.path.isdir(self.sort_data_dir):
            os.makedirs(self.sort_data_dir)
        for i in temp.index:
            df_temp = df.loc[df[' Label'] == i]
            df_temp.to_csv(self.sort_data_dir + '/CIC_IDS_2017_' + i + '.csv', index=False)


    def load_raw_data(self):
        pass
        # [{} for i in range(len())]
        # for i in

        # self.data_array = np.array(df)
        # print('type(self.data_array): ', type(self.data_array))
        # print('self.data_array[0]: ', self.data_array[0])
        # print('self.data_array.shape: ', self.data_array.shape)
        # print('self.data_array.dtype: ', self.data_array.dtype)

        # self.data_array = np.loadtxt(self.raw_data_dir_list[0], dtype=np.string_, delimiter=',')[1:]
        # print('i = 0, new_data_array: ', self.data_array.shape[0])
        # for i in range(1, len(self.raw_data_dir_list)):
        #     new_data_array = np.loadtxt(self.raw_data_dir_list[i], dtype=np.string_, delimiter=',')[1:]
        #     print('i = %d, new_data_array: '%i, new_data_array.shape[0])
        #     self.data_array = np.vstack((self.data_array, new_data_array))
        # self.data_num = self.data_array.shape[0]
        # print('### load raw data over ###')

    
    def load_processed_data(self):
        if self.mode == 'Train':
            self.data_array = np.loadtxt(self.processed_train_data_dir, dtype=np.float32, delimiter=',')
        if self.mode == 'Test':
            self.data_array = np.loadtxt(self.processed_test_data_dir, dtype=np.float32, delimiter=',')
        self.data_num = self.data_array.shape[0]


    def __getitem__(self, index):
        return self.data_array[index,:41], self.data_array[index, 41]


    def __len__(self):
        return self.data_array.shape[0]


    def save_data(self):
        num = int(self.data_num * 0.6)
        np.savetxt(self.processed_train_data_dir, self.data_array[:num,:], delimiter=',')
        np.savetxt(self.processed_test_data_dir, self.data_array[num:,:], delimiter=',')


# class KDD_CUP_99_DataLoader(DataLoader):
#     def __init__(self, root_dir, batch_size=1, mode='Train', category=None):
#         assert mode == 'Train' or mode == 'Test'
#         self.data = KDD_CUP_99_Base(root_dir, mode)
#         self.data.load_processed_data()
#         if category != None and category in self.data.label_category.keys():
#             self.data.get_category_data(category)
#         super().__init__(self.data, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    dataset = CIC_IDS_2107_Base('E:/DataSets/CIC-IDS2017')
    dataset.data_sort()
    # dataset.load_raw_data()
    # dataset.data_process()
    # for i in range(10):
    #     print(dataset[i])
    # print('data length: ', dataset.__len__())
    # print(dataset.to_num_column_dic)
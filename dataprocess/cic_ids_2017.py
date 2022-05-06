import numpy as np
import pandas as pd
import glob
import os
from torch.utils.data import DataLoader, Dataset

pd.set_option('display.max_columns', None)


class CIC_IDS_2107_Base(Dataset):

    def __init__(self, root_dir, mode='Train'):
        assert mode == 'Train' or mode == 'Test'
        self.mode = mode
        self.data_array = None
        self.pd_DataFrame = None

        self.original_columns_name = [
            ' Destination Port', ' Protocol', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
            'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max', ' Fwd Packet Length Min',
            ' Fwd Packet Length Mean', ' Fwd Packet Length Std', 'Bwd Packet Length Max', ' Bwd Packet Length Min',
            ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean',
            ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
            ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags',
            ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s',
            ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std',
            ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
            ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio',
            ' Average Packet Size', ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk',
            ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
            'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward',
            ' Init_Win_bytes_backward', ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean', ' Active Std',
            ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min', ' Label'
        ]
        self.columns_keep_idx = [
            4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64,
            65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84
        ]
        self.Dports_use = [80, 53, 443, 21, 22]
        self.data_num = 0
        # self.label_category = ['BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye', 'FTP-Patator', 'SSH-Patator', 'DoS slowloris', 'DoS Slowhttptest', 'Bot', 'Web Attack Brute Force', 'Web Attack XSS', 'Infiltration', 'Web Attack Sql Injection', 'Heartbleed']
        self.label_category = [
            'BENIGN', 'Dos', 'PortScan', 'DDoS', 'Brute Force', 'Bot', 'Web Attack', 'Infiltration', 'Heartbleed'
        ]
        self.columns_name = [
            ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets', 'Total Length of Fwd Packets',
            ' Total Length of Bwd Packets', ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean',
            ' Fwd Packet Length Std', 'Bwd Packet Length Max', ' Bwd Packet Length Min', ' Bwd Packet Length Mean',
            ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max',
            ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total',
            ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
            ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
            ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
            'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count',
            ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size',
            ' Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk',
            ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets',
            ' Subflow Bwd Bytes', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd',
            ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std',
            ' Idle Max', ' Idle Min', ' Destination Port_0', ' Destination Port_21', ' Destination Port_22',
            ' Destination Port_53', ' Destination Port_80', ' Destination Port_443', ' Protocol_0', ' Protocol_6',
            ' Protocol_17', ' Label_BENIGN', ' Label_Bot', ' Label_Brute Force', ' Label_DDoS', ' Label_Dos',
            ' Label_Heartbleed', ' Label_Infiltration', ' Label_PortScan', ' Label_Web Attack'
        ]

        self.raw_data_dir = root_dir + '/raw_data/'
        self.sort_data_dir = root_dir + '/sorted_data/'
        self.processed_train_data_dir = root_dir + '/cic_ids_2017_trainset.csv'
        self.processed_test_data_dir = root_dir + '/cic_ids_2017_testset.csv'

    def get_category_data(self, category):
        indices = self.label_category[category]
        select = self.__if_in(self.data_array, indices)

        self.data_array = self.data_array[select]

    def data_process(self):
        # here to make columns ' name more concise
        self.pd_DataFrame[' Protocol'] = self.pd_DataFrame[' Protocol'].astype(int)
        # make Destination ports more useful
        Dport_replace = lambda x: x if x in self.Dports_use else 0
        self.pd_DataFrame[' Destination Port'] = self.pd_DataFrame[' Destination Port'].astype(int)
        self.pd_DataFrame[' Destination Port'] = self.pd_DataFrame[' Destination Port'].apply(Dport_replace)
        # use one-hot coding the ['Protocol'] and data_label
        self.pd_DataFrame = pd.get_dummies(self.pd_DataFrame,
                                           columns=[' Destination Port', ' Protocol', ' Label'],
                                           prefix=[' Destination Port', ' Protocol', ' Label'])
        # make the sequence of data random
        self.pd_DataFrame = self.pd_DataFrame.sample(frac=1, random_state=1)

        columns = self.pd_DataFrame.columns.to_list()
        self.columns_name = columns

        for i in range(len(columns)):
            col_max = self.pd_DataFrame[columns[i]].max()
            col_min = self.pd_DataFrame[columns[i]].min()
            if col_max == col_min:
                self.pd_DataFrame[columns[i]] = 0
            else:
                self.pd_DataFrame[columns[i]] = (self.pd_DataFrame[columns[i]] - col_min) / (col_max - col_min)

    def data_sort(self):
        raw_data_dir_list = glob.glob(os.path.join(self.raw_data_dir, '*'))

        df = pd.read_csv(raw_data_dir_list[0], usecols=self.columns_keep_idx, encoding='unicode_escape', dtype={' Label': str})
        for i in range(1, 8):
            _df = pd.read_csv(raw_data_dir_list[i],
                              usecols=self.columns_keep_idx,
                              encoding='unicode_escape',
                              dtype={' Label': str})
            df = df.append(_df)

        df = df.replace(np.inf, np.nan).replace(-np.inf, np.nan).replace('', np.nan).dropna()
        df = df.sample(frac=1)

        df[' Label'].replace(['DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest'], 'Dos', inplace=True)
        df[' Label'].replace(['Web Attack \x96 Brute Force', 'Web Attack \x96 XSS', 'Web Attack \x96 Sql Injection'],
                             'Web Attack',
                             inplace=True)
        df[' Label'].replace(['FTP-Patator', 'SSH-Patator'], 'Brute Force', inplace=True)

        labels = df[' Label'].value_counts(normalize=True)

        if not os.path.isdir(self.sort_data_dir):
            os.makedirs(self.sort_data_dir)
        for i in labels.index:
            df_temp = df.loc[df[' Label'] == i]
            df_temp.to_csv(self.sort_data_dir + '/CIC_IDS_2017_' + i + '.csv', index=False)

    def load_sorted_data(self):
        sort_data_dir_list = [self.sort_data_dir + i for i in os.listdir(self.sort_data_dir)]

        df = pd.read_csv(sort_data_dir_list[0], header=0, nrows=50000)
        for i in range(1, len(sort_data_dir_list)):
            _df = pd.read_csv(sort_data_dir_list[i], header=0, nrows=50000)
            df = df.append(_df)
        print('load sorted data over')

        self.pd_DataFrame = df
        self.data_num = self.pd_DataFrame.shape[0]

    def load_processed_data(self):
        if self.mode == 'Train':
            self.data_array = np.loadtxt(self.processed_train_data_dir, dtype=np.float32, delimiter=',')
        if self.mode == 'Test':
            self.data_array = np.loadtxt(self.processed_test_data_dir, dtype=np.float32, delimiter=',')
        self.data_num = self.data_array.shape[0]

    def save_data(self):
        num = int(self.data_num * 0.6)
        self.pd_DataFrame.iloc[:num].to_csv(self.processed_train_data_dir, index=False, header=False)
        self.pd_DataFrame.iloc[num:].to_csv(self.processed_test_data_dir, index=False, header=False)

    def __getitem__(self, index):
        return self.data_array[index, :85], self.data_array[index, 85:]

    def __len__(self):
        return self.data_array.shape[0]


class CIC_IDS_2107_DataLoader(DataLoader):

    def __init__(self, root_dir, batch_size=1, mode='Train', category=None):
        assert mode == 'Train' or mode == 'Test'
        self.data = CIC_IDS_2107_Base(root_dir, mode)
        self.data.load_processed_data()
        if category != None and category in self.data.label_category.keys():
            self.data.get_category_data(category)
        super().__init__(self.data, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    dataset = CIC_IDS_2107_DataLoader('E:/DataSets/CIC-IDS2016')
    # # dataset.data_sort()
    # dataset.load_sorted_data()
    # dataset.data_process()
    # dataset.save_data()
    # print(dataset.columns_name)
    # dataset.load_processed_data()
    cnt = 0
    for i in dataset:
        print(i)
        cnt += 1
        if cnt > 10:
            break

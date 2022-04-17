import numpy as np

from torch.utils.data import DataLoader, Dataset


class KDD_CUP_99_Base(Dataset):
    def __init__(self, root_dir, mode='Train'):
        assert mode == 'Train' or mode == 'Test'
        self.mode = mode
        self.data_array = None
        self.to_num_column_idx = [1,2,3,41]
        self.columns_name = ['duration', 'protocol', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
                            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
                            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_hot_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 
                            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate'
                            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']
        self.discrete_column = [1,2,3,11,13,14,20,21,41]
        self.data_num = 50000
        self.to_num_column_dic = None
        self.load_data_dir = root_dir + '/kddcup.data.corrected'
        self.processed_train_data_dir = root_dir + '/kddcup_trainset.csv'
        self.processed_test_data_dir = root_dir + '/kddcup_testset.csv'


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


    def kdd_cup99_numerical(self):
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
        self.data_array = self.data_array.astype(np.float32)

        return dict(zip(self.to_num_column_idx, to_num_column_dic))


    def disorder(self):
        np.random.shuffle(self.data_array)


    def data_process(self):
        self.to_num_column_dic = self.kdd_cup99_numerical()
        self.normalize()
        self.disorder()


    def load_raw_data(self):
        self.data_array = np.loadtxt(self.load_data_dir, dtype=np.string_, delimiter=',')
        self.data_num = self.data_array.shape[0]

    
    def load_processed_data(self):
        if self.mode == 'Train':
            self.data_array = np.loadtxt(self.processed_train_data_dir, dtype=np.float32, delimiter=',')
        if self.mode == 'Test':
            self.data_array = np.loadtxt(self.processed_test_data_dir, dtype=np.float32, delimiter=',')


    def __getitem__(self, index):
        return self.data_array[index,:41], self.data_array[index, 41]


    def __len__(self):
        return self.data_array.shape[0]


    def save_data(self):
        num = int(self.data_num * 0.6)
        np.savetxt(self.processed_train_data_dir, self.data_array[:num,:], delimiter=',')
        np.savetxt(self.processed_test_data_dir, self.data_array[num:,:], delimiter=',')


class KDD_CUP_99_DataLoader(DataLoader):
    def __init__(self, root_dir, batch_size=1, mode='Train'):
        assert mode == 'Train' or mode == 'Test'
        self.data = KDD_CUP_99_Base(root_dir, mode)
        self.data.load_processed_data()
        super().__init__(self.data, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    DATA_DIR = 'E:/DataSets/kddcup.data'
    dataset = KDD_CUP_99_Base(DATA_DIR)
    dataset.load_raw_data()
    dataset.data_process()
    dataset.save_data()
    print(dataset.__len__())
    print(dataset.to_num_column_dic)



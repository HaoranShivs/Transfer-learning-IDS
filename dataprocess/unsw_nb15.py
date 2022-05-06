import os
import numpy as np

from torch.utils.data import DataLoader, Dataset

class UNSW_NB15_BASE(Dataset):
    def __init__(self, root_dir, mode='Train'):
        assert mode == 'Train' or mode == 'Test'
        self.mode = mode
        self.data_array = None
        self.columns_name = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 
                            'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 
                            'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat', 
                            'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
                            'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label']
        self.to_num_column_idx = [0,2,4,5,13,47]
        self.discrete_column = [0,1,2,3,4,5,13,35,38,41,47,48]
        self.to_num_column_dic = {0: {b'59.166.0.4': 0, b'59.166.0.1': 1, b'59.166.0.5': 2, b'59.166.0.2': 3, b'59.166.0.0': 4, b'59.166.0.3': 5, b'59.166.0.9': 6, b'59.166.0.6': 7, b'59.166.0.8': 8, b'59.166.0.7': 9, b'175.45.176.1': 10, b'175.45.176.3': 11, b'175.45.176.0': 12, b'149.171.126.18': 13, b'149.171.126.15': 14, b'149.171.126.14': 15, b'175.45.176.2': 16, b'149.171.126.10': 17, b'149.171.126.1': 18, b'10.40.85.1': 19, b'10.40.182.1': 20, b'10.40.182.6': 21, b'10.40.85.30': 22, b'10.40.182.3': 23, b'10.40.170.2': 24, b'10.40.85.10': 25, b'149.171.126.5': 26, b'149.171.126.3': 27, b'149.171.126.4': 28, b'149.171.126.6': 29, b'149.171.126.2': 30, b'149.171.126.8': 31, b'149.171.126.9': 32, b'149.171.126.7': 33, b'149.171.126.0': 34, b'149.171.126.13': 35, b'192.168.241.243': 36, b'149.171.126.12': 37, b'149.171.126.11': 38, b'149.171.126.19': 39, b'149.171.126.17': 40, b'149.171.126.16': 41, b'127.0.0.1': 42}, 2: {b'149.171.126.1': 0, b'149.171.126.3': 1, b'149.171.126.2': 2, b'149.171.126.4': 3, b'149.171.126.5': 4, b'149.171.126.0': 5, b'149.171.126.9': 6, b'149.171.126.7': 7, b'149.171.126.6': 8, b'149.171.126.8': 9, b'175.45.176.3': 10, b'149.171.126.18': 11, b'175.45.176.1': 12, b'149.171.126.15': 13, b'175.45.176.0': 14, b'149.171.126.14': 15, b'149.171.126.10': 16, b'149.171.126.17': 17, b'149.171.126.12': 18, b'149.171.126.11': 19, b'149.171.126.13': 20, b'149.171.126.16': 21, b'149.171.126.19': 22, b'224.0.0.5': 23, b'10.40.182.3': 24, b'10.40.182.255': 25, b'10.40.85.1': 26, b'10.40.170.2': 27, b'10.40.85.30': 28, b'59.166.0.0': 29, b'59.166.0.1': 30, b'59.166.0.2': 31, b'59.166.0.9': 32, b'59.166.0.4': 33, b'59.166.0.6': 34, b'59.166.0.8': 35, b'59.166.0.5': 36, b'59.166.0.7': 37, b'10.40.198.10': 38, b'59.166.0.3': 39, b'192.168.241.50': 40, b'192.168.241.243': 41, b'175.45.176.2': 42, b'192.168.241.5': 43, b'224.0.0.1': 44, b'10.40.182.6': 45, b'32.50.32.66': 46, b'127.0.0.1': 47}, 
                                  4: {b'tcp': 0, b'udp': 1, b'unas': 2, b'arp': 3, b'ospf': 4, b'sctp': 5, b'icmp': 6, b'any': 7, b'gre': 8, b'rsvp': 9, b'ipv6': 10, b'pim': 11, b'swipe': 12, b'sun-nd': 13, b'mobile': 14, b'sep': 15, b'ib': 16, b'idpr': 17, b'idpr-cmtp': 18, b'idrp': 19, b'iatp': 20, b'ifmp': 21, b'i-nlsp': 22, b'igp': 23, b'il': 24, b'ipv6-route': 25, b'ip': 26, b'isis': 27, b'ipx-n-ip': 28, b'ipcomp': 29, b'ipcv': 30, b'ipip': 31, b'iplt': 32, b'ipnip': 33, b'ippc': 34, b'irtp': 35, b'ipv6-frag': 36, b'ipv6-no': 37, b'ipv6-opts': 38, b'hmp': 39, b'zero': 40, b'gmtp': 41, b'cpnx': 42, b'a/n': 43, b'aes-sp3-d': 44, b'argus': 45, b'aris': 46, b'ax.25': 47, b'bbn-rcc': 48, b'bna': 49, b'br-sat-mon': 50, b'cbt': 51, b'cftp': 52, b'chaos': 53, b'compaq-peer': 54, b'cphb': 55, b'crtp': 56, b'ggp': 57, b'crudp': 58, b'dcn': 59, b'ddp': 60, b'ddx': 61, b'dgp': 62, b'egp': 63, b'eigrp': 64, b'emcon': 65, b'encap': 66, b'etherip': 67, b'fc': 68, b'iso-tp4': 69, b'fire': 70, b'iso-ip': 71, b'leaf-2': 72, b'kryptolan': 73, b'sccopmce': 74, b'sdrp': 75, b'secure-vmtp': 76, b'skip': 77, b'sm': 78, b'smp': 79, b'snp': 80, b'sprite-rpc': 81, b'sps': 82, b'srp': 83, b'st2': 84, b'stp': 85, b'tcf': 86, b'tlsp': 87, b'tp++': 88, b'trunk-1': 89, b'trunk-2': 90, b'ttp': 91, b'uti': 92, b'vines': 93, b'visa': 94, b'vmtp': 95, b'vrrp': 96, b'wb-expak': 97, b'wb-mon': 98, b'wsn': 99, b'xnet': 100, b'xns-idp': 101, b'l2tp': 102, b'scps': 103, b'sat-mon': 104, b'sat-expak': 105, b'larp': 106, b'leaf-1': 107, b'xtp': 108, b'merit-inp': 109, b'mfe-nsp': 110, b'mhrp': 111, b'micp': 112, b'mtp': 113, b'mux': 114, b'narp': 115, b'netblt': 116, b'nsfnet-igp': 117, b'nvp': 118, b'pgm': 119, b'pipe': 120, b'pnni': 121, b'pri-enc': 122, b'prm': 123, b'ptp': 124, b'pup': 125, b'pvp': 126, b'qnx': 127, b'rdp': 128, b'rvd': 129, b'3pc': 130, b'igmp': 131, b'udt': 132, b'rtp': 133, b'esp': 134}, 
                                  5: {b'FIN': 0, b'CON': 1, b'INT': 2, b'REQ': 3, b'RST': 4, b'ECO': 5, b'CLO': 6, b'URH': 7, b'ACC': 8, b'PAR': 9, b'TST': 10, b'ECR': 11, b'no': 12, b'URN': 13, b'MAS': 14, b'TXD': 15}, 
                                  13: {b'-': 0, b'dns': 1, b'http': 2, b'ftp-data': 3, b'smtp': 4, b'ftp': 5, b'ssh': 6, b'pop3': 7, b'dhcp': 8, b'ssl': 9, b'snmp': 10, b'radius': 11, b'irc': 12}, 
                                  47: {b'Normal': 0, b'Generic': 1, b'Exploits': 2, b'Fuzzers': 3, b'DoS': 4, b' Reconnaissance': 5, b'Analysis': 6, b'Backdoors': 7, b'Shellcode': 8, b'Worms': 9}}
        
        self.label_dic = {b'Normal': 0, b'Generic': 1, b'Exploits': 2, b'Fuzzers': 3, b'DoS': 4, b' Reconnaissance': 5, b'Analysis': 6, b'Backdoors': 7, b'Shellcode': 8, b'Worms': 9}

        self.data_num = 0
        self.raw_data_dir_list = [root_dir + '/raw_dataset/' + i for i in os.listdir(root_dir + '/raw_dataset/')]

        if not os.path.isdir(root_dir + '/prepared_data'):
            os.makedirs(root_dir + '/prepared_data')
        self.processed_train_data_dir = root_dir + '/prepared_data/unsw_nb_99_trainset.csv'
        self.processed_test_data_dir = root_dir + '/prepared_data/unsw_nb_99_testset.csv'


    def load_raw_data(self):
        def del_unicode_start(x):
            if '\xef\xbb\xbf' in x:
                return x[3:]
            else:
                return x
        def null_string(x):
            if x == '':
                return '0'
            if x == ' ':
                return '0'
            else:
                return x
        def garbled(x):
            if x == '0x000b':
                return '11'
            elif x == '0x000c':
                return '12'
            elif x == '0xc0a8':
                return '49320'
            elif x == '0x20205321':
                return '0'
            elif x =='0xcc09':
                return '52233'
            elif x == '-':
                return '0'
            else:
                return x
        def label(x):
            if x  == '':
                return 'Normal'
            elif x == ' Fuzzers ' or x == ' Fuzzers':
                return 'Fuzzers'
            elif x == 'Reconnaissanc':
                return 'Reconnaissance'
            elif x == ' Reconnaissance ' or x == ' Reconnaissan':
                return 'Reconnaissance'
            elif x == 'Backdoor':
                return 'Backdoors'
            elif x == ' Shellcode ':
                return 'Shellcode'
            else:
                return x
        self.data_array = np.loadtxt(self.raw_data_dir_list[0], dtype=np.string_, delimiter=',', encoding='Latin-1', converters={0:del_unicode_start, 1:garbled, 3:garbled, 37:null_string, 38:null_string, 39:null_string, 47:label})
        for i in range(1, len(self.raw_data_dir_list)):
            self.data_array = np.vstack((self.data_array, np.loadtxt(self.raw_data_dir_list[i], dtype=np.string_, delimiter=',', encoding='Latin-1', converters={0:del_unicode_start, 1:garbled, 3:garbled, 37:null_string, 38:null_string, 39:null_string, 47:label})))
        self.data_num = self.data_array.shape[0]
        self.__unsw_nb15_label_process()


    def unsw_nb_numerical(self):
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

        self.to_num_column_dic = dict(zip(self.to_num_column_idx, to_num_column_dic))

        self.data_array = self.data_array.astype(np.float32)


    def normalize(self):
        column_idx = [i for i in range(len(self.columns_name))]
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
        self.del_redundant_label()
        

    def del_redundant_label(self):
        self.data_array = self.data_array[:,:48]


    def __unsw_nb15_label_process(self):
        index = self.data_array[:,47] == b'Reconnaissanc'
        self.data_array[index,47] = b'Reconnaissance'


    def __getitem__(self, index):
        return self.data_array[index,:47], self.data_array[index, 47]


    def __len__(self):
        return self.data_array.shape[0]


    def save_data(self):
        num = int(self.data_num * 0.6)
        np.savetxt(self.processed_train_data_dir, self.data_array[:num], delimiter=',')
        np.savetxt(self.processed_test_data_dir, self.data_array[num:], delimiter=',')

    
    def load_processed_data(self):
        if self.mode == 'Train':
            self.data_array = np.loadtxt(self.processed_train_data_dir, dtype=np.float32, delimiter=',')
        if self.mode == 'Test':
            self.data_array = np.loadtxt(self.processed_test_data_dir, dtype=np.float32, delimiter=',')
        self.data_num = self.data_array.shape[0]


class UNSW_NB15_DataLoader(DataLoader):
    def __init__(self, root_dir, batch_size=1, mode='Train', category=None):
        assert mode == 'Train' or mode == 'Test'
        self.data = UNSW_NB15_BASE(root_dir, mode)
        self.data.load_processed_data()
        if category != None and category in self.data.label_category.keys():
            self.data.get_category_data(category)
        super().__init__(self.data, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    dataset = UNSW_NB15_BASE('E:/DataSets/UNSW-NB15 - CSV Files')
    dataset.load_raw_data()
    dataset.data_process()
    dataset.save_data()
    print(dataset.to_num_column_dic)
    # dataset.load_processed_data()
    # dataset.test()

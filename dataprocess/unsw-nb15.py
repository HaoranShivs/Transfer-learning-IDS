from re import T
import numpy as np

from torch.utils.data import DataLoader, Dataset

class UNSW_NB15_BASE(Dataset):
    def __init__(self, root_dir, mode='Train'):
        assert mode == 'Train' or mode == 'Test'
        self.mode = mode
        self.data_array = None
        self.to_num_column_idx = [1,2,3,41]
        self.columns_name = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 
                            'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 
                            'res_bdy_len', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat', 
                            'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
                            'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label']
        self.discrete_column = [1,2,3,11,13,14,20,21,41]
        self.data_num = 50000
        self.to_num_column_dic = None
        self.load_data_dir = root_dir + '/kddcup.data.corrected'
        self.processed_train_data_dir = root_dir + '/kddcup_trainset.csv'
        self.processed_test_data_dir = root_dir + '/kddcup_testset.csv'
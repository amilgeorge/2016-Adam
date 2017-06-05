
from dataprovider.davis_cached import DataAccessHelper as DataAccessHelper2017
from dataprovider.davis_cached_2016 import DataAccessHelper as DataAccessHelper2016
import re
import numpy as np
import os
import skimage
import skimage.io as io
from utils import evalhelper

IMDB_DAVIS_2017 = 'davis2017'
IMDB_DAVIS_2016 = 'davis2016'
IMDB_DAVIS_COMBO = 'daviscombo'
IMDB_SEQ_DAVIS2016 = 'seqdavis2016'
IMDB_FINETUNE_DAVIS2016 = 'finetunedavis2016'
IMDB_CUSTOM_MASK_DAVIS2016 = 'custommaskdavis2016'
IMDB_CUSTOM_MASK_DAVIS2017 = 'custommaskdavis2017'

class DB2017:

    IMDB_NAME = IMDB_DAVIS_2017

    def __init__(self):
        self.davis = DataAccessHelper2017()
        self.train_infos = self.load_train_infos()

    def load_train_infos(self):
        print("preparing file list for davis 2017 imdb...")
        file_list = []
        for seq in self.davis.train_sequence_list():
            all_frames = self.davis.all_frames_nums(seq)
            num_objects = self.davis.num_objects(seq)
            for frame_no in all_frames:
                for obj_id in range(1,num_objects+1):
                    file_list.append((seq, frame_no,obj_id))

        print("created list of {} files".format(len(file_list)))
        return file_list

    def num_train_infos(self):
        return  len(self.train_infos)

    def get_at(self,index, prev_frame_no_calculator):
        seq, frame_no, obj_id  = self.train_infos[index]
        input_ch7 = self.davis.get_input_ch7_gt_mask(seq, frame_no, obj_id, prev_frame_no_calculator)
        label_path = self.davis.label_path(seq,frame_no)
        label = self.davis.read_label(label_path,obj_id)
        return input_ch7,label


class DB2016:

    IMDB_NAME = IMDB_DAVIS_2016

    def __init__(self):
        self.davis = DataAccessHelper2016()
        self.train_infos = self.load_train_infos()

    def load_train_infos(self):
        print("preparing file list for davis 2016 imdb...")
        file_list = []
        for seq in self.davis.train_sequence_list():
            all_frames = self.davis.all_frames_nums(seq)
            for frame_no in all_frames:
                file_list.append((seq, frame_no))

        print("created list of {} files".format(len(file_list)))
        return file_list

    def num_train_infos(self):
        return  len(self.train_infos)

    def get_at(self,index, prev_frame_no_calculator):
        seq, frame_no  = self.train_infos[index]
        input_ch7 = self.davis.get_input_ch7_gt_mask(seq, frame_no, prev_frame_no_calculator)
        label_path = self.davis.label_path(seq,frame_no)
        label = self.davis.read_label(label_path)
        return input_ch7,label


class CustomMaskDB2017:
    IMDB_NAME = IMDB_CUSTOM_MASK_DAVIS2017
    davis = DataAccessHelper2017()
    POLICY_CM = "custom_mask_folder"
    POLICY_SELECT_CM = "select_cm"

    class StatsSelectCM:
        def __init__(self, mask_folder):
            self.mask_folder = mask_folder
            self.gt_count = 0
            self.custom_count = 0

        def disp(self):
            return "SCM: folder: {} gt_count = {} cust_count = {} ".format(self.mask_folder, self.gt_count,
                                                                           self.custom_count)

        def increment_gt(self):
            self.gt_count += 1

        def increment_custom(self):
            self.custom_count += 1

    def __init__(self):
        self.train_infos = self.load_train_infos()
        self.policy = None
        self.policy_params = None
        self.mask_folder = None
        self.stats = None
        self.stats_hist = []

    def load_train_infos(self):
        print("preparing file list for davis 2016 imdb...")
        file_list = []
        for seq in self.davis.train_sequence_list():
            all_frames = self.davis.all_frames_nums(seq)
            for frame_no in all_frames:
                file_list.append((seq, frame_no))

        print("created list of {} files".format(len(file_list)))
        return file_list

    def set_policy(self, policy, params):
        self.policy = policy
        self.policy_params = params

    def set_mask_folder(self, folder_path):
        assert self.policy is not None, 'set policy first'
        self.mask_folder = folder_path

        if self.policy == CustomMaskDB2017.POLICY_SELECT_CM:
            if self.stats is not None:
                self.stats_hist.append(self.stats)

            self.stats = CustomMaskDB2017.StatsSelectCM(folder_path)
            print(self.disp_stats())

    def num_train_infos(self):
        return len(self.train_infos)

    def custom_label_path(self, seq, obj_id,frame_no):
        return os.path.join(self.mask_folder, seq,obj_id, '{0:05}.png'.format(frame_no))

    def read_custom_mask(self, label_path):
        label = skimage.img_as_ubyte(io.imread(label_path, as_grey=True))
        return label > 127

    def get_mask_from_folder(self, seq, obj_id,frame_no):

        if frame_no == 0:
            prev_mask_path = self.davis.label_path(seq, frame_no)
            prev_mask = np.uint8(self.davis.read_label(prev_mask_path,obj_id) * 255)
        else:
            prev_mask_path = self.custom_label_path(seq,obj_id, frame_no)
            prev_mask = self.read_custom_mask(prev_mask_path)
            assert np.logical_or((prev_mask == 1), (prev_mask == 0)).all(), "expected 0 or 1 in binary mask"
            prev_mask = np.uint8(prev_mask * 255)

        return prev_mask

    def get_selected_mask(self, seq, obj_id, frame_no):
        gt_mask_path = self.davis.label_path(seq, frame_no)

        gt_prev_mask = np.uint8(self.davis.read_label(gt_mask_path,obj_id) * 255)
        cust_prev_mask = self.get_mask_from_folder(seq,obj_id, frame_no)

        thres_J = self.policy_params
        eval_j = evalhelper.db_eval_iou(gt_prev_mask, cust_prev_mask)

        if eval_j >= thres_J:
            self.stats.increment_custom()
            return cust_prev_mask

        else:
            self.stats.increment_gt()
            return gt_prev_mask

    def get_custom_mask(self, seq,obj_id, frame_no):
        if self.policy == CustomMaskDB2016.POLICY_CM:
            return self.get_mask_from_folder(seq,obj_id, frame_no)
        elif self.policy == CustomMaskDB2016.POLICY_SELECT_CM:
            return self.get_selected_mask(seq,obj_id, frame_no)

    def get_at(self, index, prev_frame_no_calculator):

        seq, frame_no,obj_id = self.train_infos[index]
        prev_frame_no = prev_frame_no_calculator(frame_no)
        prev_mask = self.get_custom_mask(seq,obj_id, prev_frame_no)
        input_ch7 = self.davis.get_input_ch7_with_mask(seq, frame_no, prev_mask, prev_frame_no_calculator)
        label_path = self.davis.label_path(seq, frame_no)
        label = self.davis.read_label(label_path,obj_id)
        return input_ch7, label

    def disp_stats(self):
        stats_strings = [s.disp() for s in self.stats_hist]
        if self.stats is not None:
            stats_strings.append(self.stats.disp())

        return "\n".join(stats_strings)

class CustomMaskDB2016:
    IMDB_NAME = IMDB_CUSTOM_MASK_DAVIS2016
    davis = DataAccessHelper2016()
    POLICY_CM = "custom_mask_folder"
    POLICY_SELECT_CM = "select_cm"

    class StatsSelectCM:
        def __init__(self,mask_folder):
            self.mask_folder = mask_folder
            self.gt_count = 0
            self.custom_count = 0
        def disp(self):
            return "SCM: folder: {} gt_count = {} cust_count = {} ".format(self.mask_folder,self.gt_count,
                                                                           self.custom_count)
        def increment_gt(self):
            self.gt_count += 1
        def increment_custom(self):
            self.custom_count += 1


    def __init__(self):
        self.train_infos = self.load_train_infos()
        self.policy = None
        self.policy_params = None
        self.mask_folder = None
        self.stats = None
        self.stats_hist = []

    def load_train_infos(self):
        print("preparing file list for davis 2016 imdb...")
        file_list = []
        for seq in self.davis.train_sequence_list():
            all_frames = self.davis.all_frames_nums(seq)
            for frame_no in all_frames:
                file_list.append((seq, frame_no))

        print("created list of {} files".format(len(file_list)))
        return file_list

    def set_policy(self,policy,params):
        self.policy = policy
        self.policy_params = params

    def set_mask_folder(self,folder_path):
        assert self.policy is not None,'set policy first'
        self.mask_folder = folder_path

        if self.policy == CustomMaskDB2016.POLICY_SELECT_CM:
            if self.stats is not None:
                self.stats_hist.append(self.stats)

            self.stats = CustomMaskDB2016.StatsSelectCM(folder_path)
            print(self.disp_stats())

    def num_train_infos(self):
        return  len(self.train_infos)

    def custom_label_path(self,seq,frame_no):
        return os.path.join(self.mask_folder,seq,'{0:05}.png'.format(frame_no))

    def read_custom_mask(self,label_path):
        label = skimage.img_as_ubyte(io.imread(label_path, as_grey=True))
        return label>127

    def get_mask_from_folder(self,seq,frame_no):

        if frame_no == 0 :
            prev_mask_path = self.davis.label_path(seq, frame_no)
            prev_mask = np.uint8(self.davis.read_label(prev_mask_path)*255)
        else :
            prev_mask_path = self.custom_label_path(seq,frame_no)
            prev_mask = self.read_custom_mask(prev_mask_path)
            assert np.logical_or((prev_mask == 1), (prev_mask == 0)).all(), "expected 0 or 1 in binary mask"
            prev_mask =  np.uint8(prev_mask*255)

        return prev_mask

    def get_selected_mask(self,seq,frame_no):
        gt_mask_path = self.davis.label_path(seq, frame_no)

        gt_prev_mask = np.uint8(self.davis.read_label(gt_mask_path)*255)
        cust_prev_mask = self.get_mask_from_folder(seq,frame_no)

        thres_J = self.policy_params
        eval_j = evalhelper.db_eval_iou(gt_prev_mask,cust_prev_mask)

        if eval_j >= thres_J:
            self.stats.increment_custom()
            return cust_prev_mask

        else:
            self.stats.increment_gt()
            return gt_prev_mask




    def get_custom_mask(self,seq,frame_no):
        if self.policy == CustomMaskDB2016.POLICY_CM:
            return self.get_mask_from_folder(seq,frame_no)
        elif self.policy == CustomMaskDB2016.POLICY_SELECT_CM:
            return self.get_selected_mask(seq,frame_no)

    def get_at(self,index, prev_frame_no_calculator):

        seq, frame_no  = self.train_infos[index]
        prev_frame_no = prev_frame_no_calculator(frame_no)
        prev_mask = self.get_custom_mask(seq, prev_frame_no)
        input_ch7 = self.davis.get_input_ch7_with_mask(seq, frame_no, prev_mask, prev_frame_no_calculator)
        label_path = self.davis.label_path(seq,frame_no)
        label = self.davis.read_label(label_path)
        return input_ch7,label

    def disp_stats(self):
        stats_strings = [s.disp() for s in self.stats_hist]
        if self.stats is not None:
            stats_strings.append(self.stats.disp())
            
        return "\n".join(stats_strings)

class SEQDB2016:

    IMDB_NAME = IMDB_SEQ_DAVIS2016

    davis = DataAccessHelper2016()

    def __init__(self,seq):
        self.seq = seq
        self.train_infos = self.load_train_infos()

    def load_train_infos(self):
        print("preparing file list for davis 2016 imdb...")
        file_list = []
        all_frames = self.davis.all_frames_nums(self.seq)
        for frame_no in all_frames:
            file_list.append((self.seq, frame_no))

        print("created list of {} files".format(len(file_list)))
        return file_list

    def num_train_infos(self):
        return  len(self.train_infos)

    def get_at(self,index, prev_frame_no_calculator):
        seq, frame_no  = self.train_infos[index]
        input_ch7 = self.davis.get_input_ch7_gt_mask(seq, frame_no, prev_frame_no_calculator)
        label_path = self.davis.label_path(seq,frame_no)
        label = self.davis.read_label(label_path)
        return input_ch7,label

class FineTuneDB2016:

    IMDB_NAME = IMDB_FINETUNE_DAVIS2016
    davis = DataAccessHelper2016()

    def __init__(self,seq):
        self.seq = seq
        self.train_infos = self.load_train_infos()

    def load_train_infos(self):
        print("preparing file list for davis 2016 imdb...")
        file_list = []
        all_frames = [0]
        for frame_no in all_frames:
            file_list.append((self.seq, frame_no))

        print("created list of {} files".format(len(file_list)))
        return file_list

    def num_train_infos(self):
        return  len(self.train_infos)

    def get_at(self,index, prev_frame_no_calculator):
        seq, frame_no  = self.train_infos[index]
        input_ch7 = self.davis.get_input_ch7_gt_mask(seq, frame_no, prev_frame_no_calculator)
        label_path = self.davis.label_path(seq,frame_no)
        label = self.davis.read_label(label_path)
        return input_ch7,label

class DBCombo:

    IMDB_NAME = IMDB_DAVIS_COMBO

    def __init__(self):

        self.db2016 = DB2016()
        self.db2017 = DB2017()

        self.train_infos = self.load_train_infos()
        self.offset_len = len(self.db2017.train_infos)

    def load_train_infos(self):
        print("preparing file list for combo imdb...")
        file_list = self.db2017.train_infos + self.db2016.train_infos
        #self.offset_len = len(self.db2017.train_infos)
        return file_list

    def num_train_infos(self):
        return  len(self.train_infos)

    def get_at(self,index, prev_frame_no_calculator):
        if index < self.offset_len:
            return self.db2017.get_at(index,prev_frame_no_calculator)
        else:
            return self.db2016.get_at(index - self.offset_len, prev_frame_no_calculator)

def get_davis_2017_imdb():
    db = DB2017()
    return db

def get_davis_2016_imdb():
    db = DB2016()
    return db

def get_seq_davis_2016_imdb(seq):
    db = SEQDB2016(seq)
    return db

def get_davis_combo_imdb():
    db = DBCombo()
    return db

def get_fintune_davis2016_imdb(seq):
    db = FineTuneDB2016(seq)
    return db

def get_custom_mask_davis2016():
    db = CustomMaskDB2016()
    return db


def get_imdb(dbname):
    if dbname == IMDB_DAVIS_2017:
        return get_davis_2017_imdb()
    elif dbname == IMDB_DAVIS_2016:
        return get_davis_2016_imdb()
    elif dbname == IMDB_DAVIS_COMBO:
        return get_davis_combo_imdb()
    elif dbname == IMDB_CUSTOM_MASK_DAVIS2016:
        return get_custom_mask_davis2016()
    elif dbname.startswith(IMDB_SEQ_DAVIS2016):
        p = re.split(',', dbname)
        return get_seq_davis_2016_imdb(p[1])
    elif dbname.startswith(IMDB_FINETUNE_DAVIS2016):
        p = re.split(',',dbname)
        return get_fintune_davis2016_imdb(p[1])




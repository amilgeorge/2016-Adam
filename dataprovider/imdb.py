
from dataprovider.davis_cached import DataAccessHelper as DataAccessHelper2017
from dataprovider.davis_cached_2016 import DataAccessHelper as DataAccessHelper2016

IMDB_DAVIS_2017 = 'davis2017'
IMDB_DAVIS_2016 = 'davis2016'
IMDB_DAVIS_COMBO = 'daviscombo'

class DB2017:

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



class DBCombo:

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


def get_davis_combo_imdb():
    db = DBCombo()
    return db

def get_imdb(dbname):
    if dbname == IMDB_DAVIS_2017:
        return get_davis_2017_imdb()
    elif dbname == IMDB_DAVIS_2016:
        return get_davis_2016_imdb()
    elif dbname == IMDB_DAVIS_COMBO:
        return get_davis_combo_imdb()



from mwptoolkit.utils.utils import read_json_data
class AbstractDataset(object):
    def __init__(self,config):
        super().__init__()
        self.validset_divide=config["validset_divide"]
        self.dataset_path=config["dataset_path"]
        self.min_word_keep=config["min_word_keep"]
        self.mask_symbol=config["mask_symbol"]
        self.symbol_for_tree=config["symbol_for_tree"]
        self.share_vocab=config["share_vocab"]
    
    def _load_dataset(self):
        trainset_file=self.dataset_path+"/trainset.json"
        validset_file=self.dataset_path+"/validset.json"
        testset_file=self.dataset_path+"/testset.json"
        self.trainset=read_json_data(trainset_file)
        self.validset=read_json_data(validset_file)
        self.testset=read_json_data(testset_file)
        if self.validset_divide != True:
            self.testset=self.testset+self.validset
            self.validset=[]
    def fix_process(self,fix):
        if fix != None:
            for idx,data in enumerate(self.trainset):
                self.trainset[idx]["equation"]=fix(data["equation"])
            for idx,data in enumerate(self.validset):
                self.validset[idx]["equation"]=fix(data["equation"])
            for idx,data in enumerate(self.testset):
                self.testset[idx]["equation"]=fix(data["equation"])
    def _preprocess(self):
        raise NotImplementedError

    def _build_vocab(self):
        raise NotImplementedError

    
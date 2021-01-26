from mwptoolkit.utils.utils import read_json_data
class AbstractDataset(object):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.dataset_path=config["dataset_path"]
    
    def _load_dataset(self):
        trainset_file=self.dataset_path+"/trainset.json"
        validset_file=self.dataset_path+"/validset.json"
        testset_file=self.dataset_path+"/testset.json"
        self.trainset=read_json_data(trainset_file)
        self.validset=read_json_data(validset_file)
        self.testset=read_json_data(testset_file)
        if self.config["validset_divide"] != True:
            self.testset=self.testset+self.validset
            self.validset=[]
    
    def _preprocess(self):
        raise NotImplementedError

    def _build_vocab(self):
        raise NotImplementedError

    
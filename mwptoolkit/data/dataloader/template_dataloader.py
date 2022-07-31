# -*- encoding: utf-8 -*-
# @Author: Yihuai Lan
# @Time: 2021/08/18 11:35:57
# @File: template_dataloader.py


from mwptoolkit.data.dataloader.abstract_dataloader import AbstractDataLoader

class TemplateDataLoader(AbstractDataLoader):
    """template dataloader.

    you need implement:

    TemplateDataLoader.__init_batches()

    We replace abstract method TemplateDataLoader.load_batch() with TemplateDataLoader.__init_batches() after version 0.0.5 .
    Their functions are similar.
    
    """
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.trainset_nums = len(dataset.trainset)
        self.validset_nums = len(dataset.validset)
        self.testset_nums = len(dataset.testset)

    def load_data(self,type:str):
        """
        Load batches, return every batch data in a generator object.

        :param type: [train | valid | test], data type.
        :return: Generator[dict], batches
        """
        if type == "train":
            self.__trainset_batch_idx=-1
            for batch in self.trainset_batches:
                self.__trainset_batch_idx = (self.__trainset_batch_idx + 1) % self.trainset_batch_nums
                yield batch
        elif type == "valid":
            self.__validset_batch_idx=-1
            for batch in self.validset_batches:
                self.__validset_batch_idx = (self.__validset_batch_idx + 1) % self.validset_batch_nums
                yield batch
        elif type == "test":
            self.__testset_batch_idx=-1
            for batch in self.testset_batches:
                self.__testset_batch_idx = (self.__testset_batch_idx + 1) % self.testset_batch_nums
                yield batch
        else:
            raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))

    def load_next_batch(self,type:str):
        """
        Return next batch data
        :param type: [train | valid | test], data type.
        :return: batch data
        """
        if type == "train":
            self.__trainset_batch_idx=(self.__trainset_batch_idx+1)%self.trainset_batch_nums
            return self.trainset_batches[self.__trainset_batch_idx]
        elif type == "valid":
            self.__validset_batch_idx = (self.__validset_batch_idx + 1) % self.validset_batch_nums
            return self.validset_batches[self.__validset_batch_idx]
        elif type == "test":
            self.__testset_batch_idx = (self.__testset_batch_idx + 1) % self.testset_batch_nums
            return self.testset_batches[self.__testset_batch_idx]
        else:
            raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))

    def init_batches(self):
        """
        Initialize batches of trainset, validset and testset.
        :return: None
        """
        self.__init_batches()

    def __init_batches(self):
        """
        In this function, you need to implement the codes of initializing batches.

        Specifically, you need to

        1. reset the list variables TemplateDataLoader.trainset_batches, TemplateDataLoader.validset_batches and TemplateDataLoader.testset_batches.
        And save corresponding every batch data in them. What value every batch includes is designed by you.

        2. reset the integer variables TemplateDataLoader.__trainset_batch_idx, TemplateDataLoader.__validset_batch_idx and TemplateDataLoader.__testset_batch_idx as -1.

        3. reset the integer variables TemplateDataLoader.trainset_batch_nums, TemplateDataLoader.validset_batch_nums and TemplateDataLoader.testset_batch_nums.
        Their values should equal to corresponding length of batches.
        """
        raise NotImplementedError

    # def load_data(self, type):
    #     if type == "train":
    #         datas = self.dataset.trainset
    #         batch_size = self.train_batch_size
    #     elif type == "valid":
    #         datas = self.dataset.validset
    #         batch_size = self.test_batch_size
    #     elif type == "test":
    #         datas = self.dataset.testset
    #         batch_size = self.test_batch_size
    #     else:
    #         raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))
    #
    #     num_total = len(datas)
    #     batch_num = int(num_total / batch_size) + 1
    #     for batch_i in range(batch_num):
    #         start_idx = batch_i * batch_size
    #         end_idx = (batch_i + 1) * batch_size
    #         if end_idx <= num_total:
    #             batch_data = datas[start_idx:end_idx]
    #         else:
    #             batch_data = datas[start_idx:num_total]
    #         if batch_data != []:
    #             batch_data = self.load_batch(batch_data)
    #             yield batch_data
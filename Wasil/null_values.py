# import necessary libraries
import numpy as np
import infinity


# check does the dataset has Null values
class NullFeatures:
    def __init__(self, dataset):
        self.last_feature = dataset.columns[-1]  # get the last feature
        self.dataset = dataset
        self.object = infinity.Infinity(self.dataset)

    def null_val(self):
        '''
        This function is going to check does the entrance dataset has null values
        if so then drop the last feature for just checking the null values not for every time.
        '''

        null_data = np.array(self.dataset.drop(self.last_feature, axis=1).isna().sum())  # check total null values
        if null_data.all() == 0:
            return self.object.infinity_to_mean()
        else:
            self.fill_null()
            return self.object.infinity_to_mean()

    def fill_null(self):
        '''
        This function is going to fill null values in any feature if it has
        '''

        # search for null values
        for feature in self.dataset.columns[:-1]:
            if self.dataset[feature].isna().sum() > 0:  # Exclude the last feature
                # fill null values with mean of its feature
                self.dataset[feature].fillna(self.dataset[feature].mean(), inplace=True)

        return self.dataset


if __name__ == '__main__':
    pass

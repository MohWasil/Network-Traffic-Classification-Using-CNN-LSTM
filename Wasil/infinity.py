# import libraries
import pandas as pd
import numpy as np


# define infinily class
class Infinity:
    def __init__(self, dataset):
        self.dataset = dataset

    def object_to_float(self):
        '''
        This function is going to convert types of features if they are object type
        '''
        
        for column in self.dataset.columns[:-1]:  # Goes to all feature except the last feature
            if self.dataset[column].dtype == 'object':  # check types of feature
                try:
                    # convert features with object type to float32
                    self.dataset[column] = pd.to_numeric(self.dataset[column], errors='coerce').astype('float32')
                except ValueError as e:
                    return e
                    # print(f"Error converting column {column} to float32: {e}")

    def infinity_to_mean(self):
        '''
        This function converts infinity values to the mean of their corresponding labels in each feature.
        '''

        # Check feature types
        self.object_to_float()

        # Get last feature as Label
        label_column = self.dataset.columns[-1]

        # Get unique labels
        unique_labels = self.dataset[label_column].unique()

        # Iterate over each label
        for label in unique_labels:
            # Filter data for the current label
            label_data = self.dataset[self.dataset[label_column] == label]

            # Iterate over each column except the label column
            for col in self.dataset.columns:
                if col == label_column:
                    continue

                # Replace infinities with NaNs to calculate the mean
                label_data[col] = label_data[col].replace([np.inf, -np.inf], np.nan)

                # Calculate the mean for the current feature and label
                mean_value = label_data[col].mean()

                # Replace infinities in the original data with the mean value
                self.dataset.loc[self.dataset[label_column] == label, col] = self.dataset.loc[self.dataset[label_column] == label, col].replace(
                    [np.inf, -np.inf], mean_value)

        return self.dataset


if __name__ == '__main__':
    pass


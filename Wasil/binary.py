# libraries
from sklearn.preprocessing import RobustScaler
from rapidfuzz import process



class Preprocess:
    def __init__(self, dataset):
        self.dataset = dataset
        self.low_val_feature = None
        self.outliers = None
        
        
        
    def standardize_feature_names(self):
        """
        Convert feature names to lowercase and remove spaces, dots, and other non-alphanumeric characters.
        """
        self.dataset.columns = [''.join(e for e in col.lower() if e.isalnum()) for col in self.dataset.columns]
        
    # Feature checking and mapping
    def feature_check(self):
        """
        Check if the dataset features match the main dataset features.
        If not, map the features using the provided mapping dictionary.
        """
        # Dropping features list
        drop_features = []
        
        # Main features name (standardized)
        main_feature = [
            'destinationport', 'flowduration', 'totalfwdpackets',
            'totalbackwardpackets', 'totallengthoffwdpackets',
            'totallengthofbwdpackets', 'fwdpacketlengthmax',
            'fwdpacketlengthmin', 'fwdpacketlengthmean',
            'fwdpacketlengthstd', 'bwdpacketlengthmax',
            'bwdpacketlengthmin', 'bwdpacketlengthmean',
            'bwdpacketlengthstd', 'flowbytess', 'flowpacketss',
            'flowiatmean', 'flowiatstd', 'flowiatmax', 'flowiatmin',
            'fwdiattotal', 'fwdiatmean', 'fwdiatstd', 'fwdiatmax',
            'fwdiatmin', 'bwdiattotal', 'bwdiatmean', 'bwdiatstd',
            'bwdiatmax', 'bwdiatmin', 'fwdpshflags', 'bwdpshflags',
            'fwdurgflags', 'bwdurgflags', 'fwdheaderlength',
            'bwdheaderlength', 'fwdpacketss', 'bwdpacketss',
            'minpacketlength', 'maxpacketlength', 'packetlengthmean',
            'packetlengthstd', 'packetlengthvariance', 'finflagcount',
            'synflagcount', 'rstflagcount', 'pshflagcount',
            'ackflagcount', 'urgflagcount', 'cweflagcount',
            'eceflagcount', 'downupratio', 'averagepacketsize',
            'avgfwdsegmentsize', 'avgbwdsegmentsize', 'fwdavgbytesbulk',
            'fwdavgpacketsbulk', 'fwdavgbulkrate', 'bwdavgbytesbulk',
            'bwdavgpacketsbulk', 'bwdavgbulkrate', 'subflowfwdpackets',
            'subflowfwdbytes', 'subflowbwdpackets', 'subflowbwdbytes',
            'initwinbytesforward', 'initwinbytesbackward',
            'actdatapktfwd', 'minsegsizemin', 'activemean',
            'activestd', 'activemax', 'activemin', 'idlemean', 'idlestd',
            'idlemax', 'idlemin', 'l7protocol'
        ]
        
        # Feature mapping (standardized)
        feature_mapping = {
            'dstport': 'destinationport', 'flowduration': 'flowduration',
            'totfwdpkts': 'totalfwdpackets', 'totbwdpkts': 'totalbackwardpackets',
            'totlenfwdpkts': 'totallengthoffwdpackets', 'totlenbwdpkts': 'totallengthofbwdpackets',
            'fwdpktlenmax': 'fwdpacketlengthmax', 'fwdpktlenmin': 'fwdpacketlengthmin',
            'fwdpktlenmean': 'fwdpacketlengthmean', 'fwdpktlenstd': 'fwdpacketlengthstd',
            'bwdpktlenmax': 'bwdpacketlengthmax', 'bwdpktlenmin': 'bwdpacketlengthmin',
            'bwdpktlenmean': 'bwdpacketlengthmean', 'bwdpktlenstd': 'bwdpacketlengthstd',
            'flowbytss': 'flowbytess', 'flowpktss': 'flowpacketss', 
            'flowiatmean': 'flowiatmean', 'flowiatstd': 'flowiatstd', 
            'flowiatmax': 'flowiatmax', 'flowiatmin': 'flowiatmin',
            'fwdiattot': 'fwdiattotal', 'fwdiatmean': 'fwdiatmean', 
            'fwdiatstd': 'fwdiatstd', 'fwdiatmax': 'fwdiatmax', 'fwdiatmin': 'fwdiatmin',
            'bwdiattot': 'bwdiattotal', 'bwdiatmean': 'bwdiatmean', 
            'bwdiatstd': 'bwdiatstd', 'bwdiatmax': 'bwdiatmax', 'bwdiatmin': 'bwdiatmin',
            'fwdpshflags': 'fwdpshflags', 'bwdpshflags': 'bwdpshflags', 
            'fwdurgflags': 'fwdurgflags', 'bwdurgflags': 'bwdurgflags', 
            'fwdheaderlen': 'fwdheaderlength', 'bwdheaderlen': 'bwdheaderlength',
            'fwdpktss': 'fwdpacketss', 'bwdpktss': 'bwdpacketss',
            'pktlenmin': 'minpacketlength', 'pktlenmax': 'maxpacketlength', 
            'pktlenmean': 'packetlengthmean', 'pktlenstd': 'packetlengthstd', 
            'pktlenvar': 'packetlengthvariance', 'finflagcnt': 'finflagcount',
            'synflagcnt': 'synflagcount', 'rstflagcnt': 'rstflagcount', 
            'pshflagcnt': 'pshflagcount', 'ackflagcnt': 'ackflagcount', 
            'urgflagcnt': 'urgflagcount', 'cweflagcnt': 'cweflagcount',
            'eceflagcnt': 'eceflagcount', 'downupratio': 'downupratio',
            'pktsizeavg': 'averagepacketsize', 'fwdsegsizeavg': 'avgfwdsegmentsize', 
            'bwdsegsizeavg': 'avgbwdsegmentsize', 'fwdbytsbavg': 'fwdavgbytesbulk', 
            'fwdpktsbavg': 'fwdavgpacketsbulk', 'fwdblkrateavg': 'fwdavgbulkrate', 
            'bwdbytsbavg': 'bwdavgbytesbulk', 'bwdpktsbavg': 'bwdavgpacketsbulk', 
            'bwdblkrateavg': 'bwdavgbulkrate', 'subflowfwdpkts': 'subflowfwdpackets',
            'subflowfwdbyts': 'subflowfwdbytes', 'subflowbwdpkts': 'subflowbwdpackets', 
            'subflowbwdbyts': 'subflowbwdbytes', 'initfwdwinbyts': 'initwinbytesforward', 
            'initbwdwinbyts': 'initwinbytesbackward', 'fwdactdatapkts': 'actdatapktfwd', 
            'fwdsegsizemin': 'minsegsizemin', 'activemean': 'activemean', 
            'activestd': 'activestd', 'activemax': 'activemax', 'activemin': 'activemin', 
            'idlemean': 'idlemean', 'idlestd': 'idlestd', 'idlemax': 'idlemax', 'idlemin': 'idlemin',
            'l7protocol': 'l7protocol'
        }
        
        
        print(self.dataset.columns)
        droped = []
        # If the dataset already has the correct features, return it as is
        if set(self.dataset.columns) == set(main_feature):
            return self.dataset
         
        # Otherwise, map the features
        if len(self.dataset.columns) == len(main_feature):
            for column in self.dataset.columns:
                if column in main_feature:
                    continue
                else:
                    droped.append(column)
            
            if droped:
                self.dataset.drop(droped, axis=1, inplace=True)
            else:
                return self.dataset
            
        elif len(self.dataset.columns) > len(main_feature):
            for column in self.dataset.columns:
                if column in main_feature:
                    continue
                elif column in feature_mapping:
                    continue
                else:
                    droped.append(column)
                    
            if droped:
                self.dataset.drop(droped, axis=1, inplace=True)
            return self.dataset
        
        aligned_columns = []
        for feature in self.dataset.columns:
            # Use fuzzy matching to find the best match if the feature isn't directly in the mapping
            best_match = process.extractOne(feature, feature_mapping.keys(), score_cutoff=80)
            if best_match:
                aligned_columns.append(feature_mapping[best_match[0]])
            else:
                drop_features.append(feature)
        # Drop unmatched features
        if drop_features:
            self.dataset.drop(drop_features, axis=1, inplace=True)
        
        # Apply the aligned columns back to the dataset
        self.dataset.columns = aligned_columns
        return self.dataset
    
    
    # Check statistic of feature values
    def statistical_analysis(self,features):
        '''
        This function is going to check Minimum, Maximum, Mean, Median, Mode, Variance and Standard Deviacion.
        '''

        # store feature name as key and statistic as values
        feat_chrac = {}
        print(self.dataset.columns)
        print(len(self.dataset.columns))
        # check each feature and store its values
        for feature in features:
            feat_chrac[feature] = [self.dataset[feature].min(), self.dataset[feature].max(),
                                self.dataset[feature].mean(), self.dataset[feature].median(),
                                self.dataset[feature].mode().iloc[0], self.dataset[feature].var(),
                                self.dataset[feature].std()]
        return feat_chrac

    # display features which have sum of less then 2 from chracteristic_feature function
    def max_two_feature(self, features):
        '''
        This function is going to check if there is flags counts and
        features with maximum values less then or equal to value 2 take them as
        no normalization.
        '''
        feature_values = []
        for key, val in self.statistical_analysis(features).items(): # taking just numerical type
            if max(val) <= 2 or 'flag' in key or 'destinationport' in key or 'l7protocol' in key:
                feature_values.append(key)
    
        return feature_values
    
    
    # Create a function for identifying any features with Outliers
    def outlier_feature(self, features):
        self.low_val_feature = self.max_two_feature(features)
        
        '''
        This function will return list of all features which has Outliers either minimum of maximum
        '''
        # storing features which has outliers (upper or lower)
        outliers = []


        for feature in features:
            Q1 = self.dataset[feature].quantile(0.25) # quantile function will return any feature value as given its percentage
            Q3 = self.dataset[feature].quantile(0.75)
            IQR = Q3 - Q1

            # identify outliers
            threshold = 1.5
            lower_boundary = Q1 - threshold * IQR # finding the lowest value of range of values in a feature
            upper_boundary = Q3 + threshold * IQR # find highest value of range of values in a feature
            low_and_high_outlier = [lower_boundary, upper_boundary]

            # Minimum & Maximum value of a feature
            min_val = self.dataset[feature].min()
            max_val = self.dataset[feature].max()


            # Checking the outliers
            if min_val < low_and_high_outlier[0]: # check if there is a value less then minimum of outliers
                if feature not in outliers and feature not in self.low_val_feature: # checking for duplicate key in dictionary and make sure of not adding no scale feature
                    outliers.append(feature)

            elif max_val > low_and_high_outlier[1]: # checking if there is a vlue higher then maximum outliers
                if feature not in outliers and feature not in self.low_val_feature: # check for duplicate key
                    outliers.append(feature)
        
        self.outliers = outliers
        # return outliers
    
    
    # Scale function
    def scale_feature(self):
        self.standardize_feature_names()
        self.feature_check()
        self.outlier_feature(self.dataset.columns)
        print(len(self.dataset.columns))
        print(self.dataset.columns)
        '''
        This function is going to scale features in two steps:
        1. RobustScaler
        '''
        # RobustScaler
        robust = RobustScaler()
        # scaling
        self.dataset[self.outliers] = robust.fit_transform(self.dataset[self.outliers])
        
        return self.dataset

    # def split_data(self):
    #     '''
    #     Spliting the dataset into x (independent) & y (dependent) variables
    #     '''
    #     x = self.dataset.drop(' Label',axis=1)
    #     y = self.dataset[' Label']
        
    #     # convert the x and y into tensor float32
    #     x = tf.convert_to_tensor(x, dtype=tf.float32)
    #     y = tf.convert_to_tensor(y, dtype=tf.float32)

    #     return x, y
    
    
    # def prediction(self):
    #     # x and y
    #     x, y = self.scale_feature()
        
    #     # load the model
    #     model = tf.keras.models.load_model('D:\Python_projects\pythonProject1\models\binary.h5')
        
    #     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    #     # Reshape to 3D
    #     x_reshaped = np.reshape(x.numpy(), (x.shape[0], x.shape[1], 1))
    
        # return model, x_reshaped, y.numpy()
    
    
    
if __name__ == "__main__":
    pass

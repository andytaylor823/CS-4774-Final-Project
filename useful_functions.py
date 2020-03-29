import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


################################################################################
#######################  Data Manipulation Functions  ##########################
################################################################################

def load_transform_split(fpath='data/ALL_YEARS_ADDED_FEATURES.csv',
                         target='rate', expand=False, split=0.1, clean=True,
                         drop_feats=['SCHOOL_YEAR','DIV_NAME','SCH_NAME','DIPLOMA_RATE'],
                         fmt='numpy',return_pipeline=False):
    '''
    Convenience function for preparing the graduation data for machine learning!
    
      INPUTS:
        fpath   - String filename of the table to load!
        target  - Type of thing to designate y_train and y_test.
                  Options:
                    'DROPOUT_RATE' - y will be dropout rates
                    'DROPOUT_N'    - y will be number of dropout students
                    'DROPOUT'      - If expand is True, y will be 0 (graduated) or 1 (dropped out),
                                      and table will be expanded into student-by-student rows.
                     None           - y will not be split off. This is for unsupervised tasks like
                                      clustering.
        expand     - Boolean whether or not to expand tables into student-by-student rows.
                      Default False
        split      - Fraction of data to split off into testing set. If 0, 1, None, or False are given,
                      data will not be split.
        clean      - Boolean whether or not to run data through a pipeline with 
                      StandardScaler and OneHot/OrdinalEncoder.
        drop_feats - Features to throw out.
        fmt        - Format of the output tables. Either 'numpy' for np.ndarray outputs or 'pandas'
                      for pandas.DataFrame outputs.
        return_pipeline - Boolean wether or not to return the pipeline used for cleaning. If clean=False,
                           None will be returned if return_pipeline=True.
      OUTPUTS:
        idk my bff jill.
    '''
    
    ### Load ###
    
    df = pd.read_csv('data/ALL_YEARS_ADDED_FEATURES.csv')
    #Get rid of any nonsense points.
    keep = (df['DROPOUT_RATE'] >= 0) & (df['DROPOUT_RATE'] <= 100)
    df = df[keep]
    #Drop unwanted features.
    if not drop_feats is None: df = df.drop(drop_feats,axis=1)
    
    
    ### Transform ###
    
    #Parse user input for predict and expand.
    if   target == 'DROPOUT_RATE':
        pass #Already stored correctly
    elif target == 'DROPOUT_N':
        df['DROPOUT_N'] = np.round(df['COHORT_CNT']*df['DROPOUT_RATE']/100.)
        df = df.drop(['DROPOUT_RATE'],axis=1)
    elif target == 'DROPOUT' and expand:
        pass #Handled in the expansion step.
    elif target is None:
        pass #Handled in splitting step.
        
    #Raise errors if input can't be used.
    elif target == 'DROPOUT' and expand:
        raise ValueError("Cannot use boolean dropout with 'expand=False.'"+\
                         "If you are certain you want to \nexpand rows (may take long),"+\
                         "rerun command with 'expand=True'")
    else:
        raise ValueError("Unrecognized value of target, %s."%(target))
        
    ### Expand ###
    # Do l8r
    if expand:
        pass #Add l8r

    
    ### Split ###
    splitting = not (split is None or split==False or split==0 or split==1)
    if not target is None: #Split X,y
        y = df[[target]]
        X = df.drop([target],axis=1)
        if splitting:        #Split Train/Test
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=split)
        else:                #No Train/Test
            X_train,y_train = X,y
    else:                  #No X,y
        X = df
        if splitting:        #Split Train/Test
            X_train,X_test = train_test_split(X,test_size=split)
        else:                #No Train/Test
            X_train = X
    
    
    ### Pipeline ###
    if clean:
        pipeline = pipeline_util(X_train,clean=False,return_pipeline=True)
        X_train = pipeline_util(X_train,pipeline=pipeline,fmt=fmt)
        if splitting: X_test = pipeline_util(X_test, pipeline=pipeline,fmt=fmt)
    
    
    ### Output Format ###
    # Do l8r
    
    ### Return! ###
    returns = [X_train]
    if splitting: returns.append(X_test)
    if not target is None: returns.append(y_train)
    if splitting and not target is None: returns.append(y_test)
    if return_pipeline: returns.append(pipeline)
        
    if len(returns) > 1:
        return tuple(returns)
    return returns[0]

def pipeline_util(X,pipeline=None,
                  clean=True,
                  fmt='numpy',
                  return_pipeline='default'):
    '''
    Convencience function for doing pipeline-related things.
    
      INPUTS:
        X               - Data to use to create pipeline or data to clean with pipeline.
        pipeline        - Pipeline to use on data. If none is provided, one is created.
                           If one is provided, this function is just a pipeline runner.
        clean           - Boolean wether or not to clean provided data. Default True.
                           if False, this function is just a pipeline maker.
        fmt             - Format of cleaned data.
                           Options:
                             'numpy' for np.ndarray output
                             'pandas' for pandas.DataFrame output
        return_pipeline - Boolean wether or not to return the pipeline itself.
                           Default follows this behavior:
                             If a pipeline is provided and clean=True, the pipeline is not returned.
      OUTPUTS:
        some or all of the following:
          pipeline - The pipeline generated or provided.
          X_clean  - X after running through the pipeline.
    '''
    if return_pipeline == 'default':
        return_pipeline = not ((not pipeline is None) and clean)
        
    if pipeline is None:
        pipeline = make_pipeline(X)
        pipeline.fit(X)

    if clean:
        X_clean = pipeline.transform(X)
        if fmt == 'numpy':
            pass #Already numpy
        elif fmt == 'pandas':
            #Get column names
            colnames = []
            for tpl in pipeline.transformers:
                transformer = tpl[1]
                try:
                    feats = transformer.get_feature_names()
                except AttributeError:
                    feats = tpl[2]
                colnames.extend(feats)
            if len(colnames) < X_clean.shape[1]:
                colroot = colnames[-1]
                colnames[-1] = colroot+'_0'
                for n in range(X_clean.shape[1] - len(colnames)):
                    colnames.append(colroot+'_%d'%(n+1))
            X_clean = pd.DataFrame(X_clean,columns=colnames)
    
    #Return!
    if return_pipeline:
        if clean:
            return X_clean,pipeline
        else:
            return pipeline
    else:
        if clean:
            return X_clean
        else:
            return
    
def make_pipeline(X,cat_thresh=10):
    '''
    Function to make a reasonable pipeline without thinking!
    
    Scheme: Features are sorted based on number of unique values, N_unique.
        If N_unique <= 2, the feature is grouped with ordinal categorical features
        If N_unique <= cat_thresh, the feature is grouped with one-hot-encoding categorical features.
        If N_unique >  cat_thresh, the feature is grouped with numerical feature and passed though standard_scaler.
    
      INPUTS:
        X          - Data to make pipeline with.
        cat_thresh - Threshold of unique values for something to be categorical/numerical.
                     N_unique <= cat_thresh -> categorical, else numerical.
      OUTPUTS:
        pipeline - The pipeline!
    '''
    #Get all features.
    all_feats = X.columns
    
    #Go through features and determine which type it is.
    num_feats = []
    ord_cat_feats = []
    ohe_cat_feats = []
    for feat in all_feats:
        unique = pd.unique(X[feat])
        if len(unique) <= 2:
            ord_cat_feats.append(feat)
        elif len(unique) <= cat_thresh:
            ohe_cat_feats.append(feat)
        else:
            num_feats.append(feat)
    
    #Make the pipeline and return!
    pipeline = ColumnTransformer([
        ('num',StandardScaler(),num_feats),
        ('ord',OrdinalEncoder(),ord_cat_feats),
        ('ohe',OneHotEncoder(categories='auto'),ohe_cat_feats),
    ])
    
    return pipeline


################################################################################
###########################  Plotting Functions  ###############################
################################################################################
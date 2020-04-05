import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import get_scorer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


################################################################################
#######################  Data Manipulation Functions  ##########################
################################################################################

def load_transform_split(fpath='data/ALL_YEARS_ADDED_FEATURES.csv',
                         target='DROPOUT_N', weight=None, split=0.1, stratsplit=None,clean=True,expand=False, 
                         drop_feats=['SCHOOL_YEAR','DIV_NAME','SCH_NAME','DIPLOMA_RATE'],
                         fmt='numpy',return_pipeline=False,random_state=None):
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
        stratsplit - Column to use for stratified splitting. ## Currently not working ##
                      If None or False if provided, stratified splitting is not performed.
                      If True is provided, target variable is used to perform stratified splitting.
                      If Column name is provided, that column is used to perform stratified splitting.
        clean      - Boolean whether or not to run data through a pipeline with 
                      StandardScaler and OneHot/OrdinalEncoder.
        drop_feats - Features to throw out.
        fmt        - Format of the output tables. Either 'numpy' for np.ndarray outputs or 'pandas'
                      for pandas.DataFrame outputs.
        return_pipeline - Boolean wether or not to return the pipeline used for cleaning. If clean=False,
                           None will be returned if return_pipeline=True.
      OUTPUTS:
        Some combo of the following (depending on what you ask for):
          X_train  - Training data
          X_test   - Testing data
          y_train  - Training labels
          y_test   - Testing labels
          pipeline - Pipeline used to clean X_train and X_test.
    '''
    
    ### Load ###
    
    df = pd.read_csv('data/ALL_YEARS_ADDED_FEATURES.csv')
    #Get rid of any nonsense points.
    keep = (df['DROPOUT_RATE'] >= 0) & (df['DROPOUT_RATE'] <= 100)
    df = df[keep]

    ### Expand ###
    # Do l8r
    if expand:
        if target=='DROPOUT':
            df = expand_table(df,taget='number')
        else:
            df = expand_table(df)

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
    elif target == 'DROPOUT' and not expand:
        raise ValueError("Cannot use boolean dropout with 'expand=False.'"+\
                         "If you are certain you want to \nexpand rows (may take long),"+\
                         "rerun command with 'expand=True'")
    else:
        raise ValueError("Unrecognized value of target, %s."%(target))
    
    ### Get weights ###
    if not weight is None: w = df[weight].astype(float)

    ### Splitting ###

    if not target is None: #Split X,y
        y = df[target]
        X = df.drop([target],axis=1)
    else:                  #No X,y
        X = df

    if stratsplit is None or stratsplit == False:
        def split_func(Z, test_size=split,random_state=random_state):
            return train_test_split(Z,test_size=split,random_state=random_state)
    else:
        if stratsplit == True:
            strat = y.to_numpy()
        else:
            strat = df[stratsplit].to_numpy()
        train_index,test_index = get_stratplit_indices(y.to_numpy(),test_size=split,random_state=random_state)
        def split_func(Z, train_index=train_index, test_index=test_index):
            try:
                return Z[train_index], Z[test_index]
            except KeyError:
                return Z.iloc[train_index],Z.iloc[test_index]

    splitting = not (split is None or split==False or split==0 or split==1)
    if splitting:        #Split Train/Test
        X_train,X_test = split_func(X)
        if not target is None: y_train,y_test = split_func(y)
        if not weight is None: w_train,w_test = split_func(w)
    else:                #No Train/Test
        X_train = X
        if not target is None: y_train = y
        if not weight is None: w_train = w
    
    ### Pipeline ###
    if clean:
        pipeline = pipeline_util(X_train,clean=False,return_pipeline=True)
        X_train = pipeline_util(X_train,pipeline=pipeline,fmt=fmt)
        if splitting: X_test = pipeline_util(X_test, pipeline=pipeline,fmt=fmt)
    else:
        pipeline = None
    
    ### Format of Output ###
    def correct_format(Z,fmt):
        if fmt == 'numpy':
            if isinstance(Z,np.ndarray):
                return Z #Already correct format
            elif isinstance(Z,pd.DataFrame) or isinstance(Z,pd.core.series.Series):
                return Z.to_numpy()
            else:
                raise TypeError("Something's gone terribly wrong. Unrecognized data format, %s"%(type(Z)))
        elif fmt == 'pandas':
            if isinstance(Z,np.ndarray):
                return pd.DataFrame(Z)
            elif isinstance(Z,pd.DataFrame) or isinstance(Z,pd.core.series.Series):
                return Z #Already correct format
            else:
                raise TypeError("Something's gone terribly wrong. Unrecognized data format, %s"%(type(Z)))
        else:
            raise ValueError("Invalid type %s, please select on of: 'numpy', 'pandas'."%(fmt))
    
    ### Return! ###
    returns = []
    returns.append(correct_format(X_train,fmt))
    if splitting: returns.append(correct_format(X_test,fmt))
    if not target is None: returns.append(correct_format(y_train,fmt))
    if splitting and not target is None: returns.append(correct_format(y_test,fmt))
    if not weight is None: returns.append(correct_format(w_train,fmt))
    if splitting and not weight is None: returns.append(correct_format(w_test,fmt))
    if return_pipeline: returns.append(pipeline)
        
    if len(returns) > 1:
        return tuple(returns)
    return returns[0]

def expand_table(df,target=None,nrows='all',progress=True):
    new_df = None
    t = time.time()
    if nrows == 'all':
        nrows = df.shape[0]
    if progress: status = simple_progress()
    for i in range(nrows):
        if progress: status.update('Row %d / %d'%(i+1,nrows))
        N_cohort = df.iloc[i]['COHORT_CNT']
        if target =='number':
            #Count up students dropping out, graduating, or otherwise
            N_drop   = np.round((df.iloc[i]['DROPOUT_RATE']/100)*N_cohort).astype(int)
            N_grad   = np.round((df.iloc[i]['DIPLOMA_RATE']/100)*N_cohort).astype(int)
            N_else   = N_cohort - N_drop - N_grad
            
            try:
                #Sanity check
                assert N_else >= 0
            except AssertionError:
                raise ValueError("Something has gone terribly wrong. Inconsistent diploma/dropout rates.")

            #Make a template row without dropout/diploma rates and with true/false instead.
            row_orig = df.iloc[i:(i+1)].copy()
            row_orig.drop(['DROPOUT_RATE','DIPLOMA_RATE'],axis=1)
            row_orig['DROP'] = 0
            row_orig['GRAD'] = 0
            
            #Make template dropout row and template graduate row.
            row_drop = row_orig.copy()
            row_drop['DROP'] = 1
            row_grad = row_orig.copy()
            row_grad['GRAD'] = 1
            
            #Make mini-table with the number of dropouts, grads, and otherwise.
            row_expanded = pd.concat( N_drop*[row_drop] + N_grad*[row_grad] + N_else*[row_orig] )
        else:
            row_orig = df.iloc[i:(i+1)].copy()
            row_expanded = pd.concat( N_cohort*[row_orig] )
        #Append mini-table to the full, expanded table.
        new_df = pd.concat([new_df,row_expanded])
        
    print((time.time()-t),'seconds')
    return new_df

def train_test_stratsplit(strat,*args,**kwargs):
    '''
    Function to perform stratified train/test splitting with similar call to sklearn's train_test_split

    INPUTS:
      strat    - List to use for stratified splitting. If continuous, this list will
                 be broken into nstrat classes with equal numbers of points per class.

      *args    - Things to split. X,y,etc.

      **kwargs - Options for get_stratsplit_indices.
                  E.g. nstrat - Number of stratifications to use.
                       test_size - Fraction to split off for test set.
                       random_state
    
    OUTPUTS:
      Z_train,Z_test for every Z in *args
    '''
    returns = []
    train_index,test_index = get_stratplit_indices(strat,**kwargs)
    for Z in args:
        try:
            Z_train = Z[train_index]
            Z_test  = Z[test_index]
        except KeyError:
            Z_train = Z.iloc[train_index]
            Z_test  = Z.iloc[test_index]
        returns.append(Z_train)
        returns.append(Z_test)
    return returns

def get_stratplit_indices(strat,nstrat=10,test_size=0.1,random_state=None,cat_thresh=20):
    '''
    Function to generate train and test indices using sklearn's StratifiedShuffleSplit

    INPUTS:
      strat     - List to use for stratified splitting. If continuous, this list will
                   be broken into nstrat classes with equal numbers of points per class.
      
      nstrat    - Number of stratifications to use.

      test_size - Fraction to split off for test set.

      random_state - Random state number to use for repeatability.

      cat_thresh - Threshold number of unique values for determining if the strat provided
                    is numerical or categorical.
    '''
    nuniq = len(np.unique(strat))
    if nuniq > cat_thresh:
        bins = [np.percentile(strat,p) for p in np.linspace(0,100,nstrat+1)]
        bins[0]-=1
        bins[-1]+=1. #Make sure bins fully encompass all datapoints.
        strat = np.digitize(strat,bins=bins)
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(split.split(strat,strat))
    return train_index, test_index

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

def color_scatter(x,y,colorby=None,ax=None,reverse=False,sortabs=False,colorbar=True,cax=None,**kwargs):
    #Define default plotting arguments
    plot_kwargs = {
        'color':'orange',
        'cmap':'cividis_r',
        'alpha':1.0,
    }
    #Update defaults with user-provided arguments
    plot_kwargs.update(kwargs)
    
    #Make axes if none were given.
    if ax is None:
        fig,ax=plt.subplots()
    
    #Determine sort and color based on inputs provided.
    if colorby is None:
        sort = np.arange(y.shape[0]).astype(int)
        color = plot_kwargs['color']
    else:
        r = 1.
        if reverse: r = -1
        if sortabs:
            sort = np.argsort(r*np.abs(colorby))
        else:
            sort = np.argsort(r*colorby)
        color = colorby[sort]
    del plot_kwargs['color']
    
    #Make scatter plot!
    scat = ax.scatter(x[sort],y[sort],c=color,**plot_kwargs)
    
    if colorbar and len(color) == len(x):
        if cax is None:
            cbar = ax.figure.colorbar(scat,ax=ax)
        else:
            cbar = ax.figure.colorbar(scat,cax=cax)
    else:
        cbar = None
    
    return ax,cbar

def plot_performance(model,name,X_train,X_test,y_train,y_test,
                     ax=None,refit=False,rmse='calc',
                     colorbar=True,cax=None,cmap='coolwarm',
                     vmin=-15,vmax=15,legend=True,fs=20,
                     random_state=42,**kwargs):

    #Make axes is necessary
    if ax is None:
        fig,ax=plt.subplots(figsize=(10,7))

    #Load data without pipeline to get un-scaled cohort counts for colorbar.
    split = X_test.shape[0]/(X_train.shape[0]+X_test.shape[0])
    tra,tes  = load_transform_split(target=None,
                                    expand=False,
                                    clean=False,
                                    split=split,
                                    return_pipeline=False,
                                    fmt='pandas',
                                    random_state=random_state)
    COHORT_CNT_tes = tes['COHORT_CNT'].to_numpy()
    COHORT_CNT_tra = tra['COHORT_CNT'].to_numpy()

    #Make sure y is flat.
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    #If you asked for it, retrain model with provided data.
    if refit and not y_train is None:
        model.fit(X_train,y_train)

    #Plot True DROPOUT_N values with pretty "danger zone" triangle.
    _plot_performance_helper(ax,**kwargs)

    #Plot training predictions for DROPOUT_N.
    ytra_pred = model.predict(X_train).flatten()
    ax,cbar = color_scatter(COHORT_CNT_tra,ytra_pred,colorby=None,color='gray',label='Training Predictions',ax=ax,**kwargs)

    #Plot testing predictions for DROPOUT_N.
    ytes_pred = model.predict(X_test).flatten()
    ytes_error = (ytes_pred-y_test)
    ax,cbar = color_scatter(COHORT_CNT_tes,ytes_pred,colorby=ytes_error,
                            cmap=cmap,ax=ax,vmin=vmin,vmax=vmax,
                            sortabs=True,label='Testing Predictions',colorbar=colorbar,cax=cax)

    #Set labels, tick fontsizes, and limits.
    ax.set_xlabel('Cohort Size',fontsize=fs)
    ax.set_ylabel('N$_{drop}$ (students)',fontsize=fs)
    if colorbar:
        cbar.ax.set_ylabel('Prediction Error (students)',fontsize=fs)
        for tlab in cbar.ax.get_yticklabels():
            tlab.set_fontsize(fs)
    for tlab in ax.get_xticklabels()+ax.get_yticklabels():
        tlab.set_fontsize(fs)
    ax.set_xlim(0,250)
    ax.set_ylim(0,100)
    
    #Get RMSE and label with model name and RMSE.
    if rmse == 'calc':
        calc_mse = get_metric('mean_squared_error')
        rmse = np.sqrt(calc_mse(model,X_test,y_test))
    ax.text(240,95,"%s\n RMSE = %.2f"%(name,rmse),fontsize=fs,horizontalalignment='right',verticalalignment='top')

    #Add legend if requested.
    if legend:
        ax.legend(loc=(0.50,0.6),fontsize=fs-3,frameon=False)
    
    #Done! Return axes in case user wants to do something else with it.
    return ax

def _plot_performance_helper(ax, **kwargs):

    #Get COHORT_CNT and DROPOUT_N for all data.
    X,DROPOUT_N = load_transform_split(split=False,target='DROPOUT_N',clean=False,fmt='pandas')
    DROPOUT_N = DROPOUT_N.to_numpy()
    COHORT_CNT = X['COHORT_CNT'].to_numpy()

    #Scatter plot DROPOUT_N vs
    ax,cbar = color_scatter(COHORT_CNT,DROPOUT_N,colorby=None,color='lightgray',label='Actual',ax=ax,zorder=0,**kwargs)

    #Plot "danger-zone" triangle (Dropout rate > 100%)
    tri = plt.Polygon([[0,0],[0,200],[200,201.5]], color='whitesmoke',zorder=0)
    ax.add_patch(tri)
    ax.plot([0,250],[0,250],ls='--',color='gray',zorder=0) # <- dashed line boarder for triangle.
    return ax

def scatter_resid(y,y_pred,colorby=None):
    #Plot (y_pred - y_test) as a function of y_test
    resid = y_pred - y
    ax,cbar = color_scatter(y,resid,colorby=colorby)
    ax.set_xlim(min(y),max(y))
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted - Actual')
    cbar.ax.set_ylabel('COHORT_CNT')

def scatter_predvreal(y,y_pred,colorby=None):
    #Plot (y_pred - y_test) as a function of y_test
    resid = y_pred - y
    ax,cbar = color_scatter(y,y_pred,colorby=colorby)
    ymi = np.min(y)
    yma = np.max(y)
    ax.plot([ymi,yma],[ymi,yma],color='black',ls='--')
    #ax.axvline(0.5*(ymi+yma),color='black',ls='--')
    ax.set_xlim(ymi,yma)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    cbar.ax.set_ylabel('COHORT_CNT')

################################################################################
##############################  Miscellaneous  #################################
################################################################################

def get_metric(scoring):
    try:
        metric = get_scorer(scoring)
    except (KeyError, ValueError) as e:
        try:
            scorer = get_scorer('neg_'+scoring)
            metric = lambda *args,**kwargs: -scorer(*args,**kwargs)
        except (KeyError, ValueError) as e:
            try:
                scorer = get_scorer(scoring[4:])
                metric = lambda *args,**kwargs: -scorer(*args,**kwargs)
            except KeyError:
                raise ValueError("Unrecognized sklearn metric")
    return metric

def preserve_state(func):
    # Decorator to prevent a function from affecting 
    #  the numpy random seed outside of its scope.
    def wrapper(*args,**kwargs):
        state = np.random.get_state() #Store the random state before function call.
        ret = func(*args,**kwargs)    #Call function.
        np.random.set_state(state)    #Revert numpy to random state from before function call.
        return ret
    return wrapper

class simple_progress:
    def __init__(self):
        self.longest = 0
    def update(self,message):
        print(self.longest*" ",end="\r") #Erase previous message.
        print(message,end="\r")          #Print message
        if len(message) > self.longest:  #Update longest message length.
            self.longest = len(message)


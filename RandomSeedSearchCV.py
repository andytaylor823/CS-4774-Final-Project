import numpy as np
import matplotlib.pyplot as plt
from time import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor

from useful_functions import simple_progress, preserve_state, get_metric

def RandomSeedSearchCV(random_model_maker,X_train,y_train,N=50,
                        validation=0.1,cv=5,scoring='mean_squared_error',
                        plot_summary=True,shield_seed=True,verbose=True,
                        random_state=None,**model_maker_kwargs):
    '''
             CAUTION: This function and many of its dependencies are in active development.
    
    Generalized random-search model tuner! Uses random seeds and user-made model-maker to search
    through "acceptable" models and find those that achieve the best cross_validation score.
    
    This function is just a shell to execute the search. It doesn't know anything about the models
    or the parameter space to search through; all of that is encapsulated in the user-provided model-making
    function.
    
    INPUTS:
      random_model_maker   - tl;dr, Function that takes integer and returns model. Returned model must implement
                                     fit and predict methods, like standard sklearn estimators do.
                                     
                             This should be a function defined by the user ahead of time. The function should
                             accept a random seed and then internally construct a model with randomly drawn 
                             hyperparameters. The idea behind this is to take responsibility of knowing the parameter
                             space and generating models within it off of the searcher and instead placing that 
                             responsibility on the user-provided model-maker. randomseed_searchCV will never know 
                             what type of model you're fitting or what range of which parameters you're sampling.
                             All it will know is that it can give random_model_maker an integer, and it will 
                             return a model that can be fit and used to predict.
                             
      **model_maker_kwargs - Any additional arguments to be passed to the model maker.
                           
      X_train,y_train      - Training data and labels.
      validation           - Fraction to split off for validation testing. Default is 0.1. If None,False,0,or 1
                              are provided, validation testing will not be done.
      cv                   - Integer N for N-fold cross validation.
      scoring              - Metric to use when ranking models.
      plot_summary         - Boolean whether or not to produce a beautiful summary plot!
      shield_seed          - Boolean whether or not to forcibly prevent the model maker from globally changing
                             the numpy random seed.
      verbose              - Boolean whether or not to show progress.
      random_state         - Random state to use for splitting off validation dataset.
    '''
    
    if verbose: progress = simple_progress()
    
    if shield_seed:
        modmkr = preserve_state(random_model_maker)
    else:
        modmkr = random_model_maker
    
    #Get metric callable from sklearn. L8r I'm gonna make it
    # so the user can provide a callable metric themselves, because I
    # don't like that sklearn uses negative mse instead of positive. It drives me bonkers.
    metric = get_metric(scoring)
    
    #Split validation set out of the training set, if 
    if validation is None or validation==False or validation>=1 or validation <=0:
        use_val = False
        Xtra,ytra = X_train,y_train
        Xval,yval = None,None
    else:
        use_val = True
        Xtra,Xval,ytra,yval = train_test_split(X_train,y_train,
                                               test_size=validation,
                                               random_state=random_state)
    
    #Draw random seeds to use for model generation.
    seeds = np.random.choice(10*N,N,replace=False)
    
    #Create empty lists to store model performance measures.
    cv_scores = np.array([])
    train_metric = np.array([])
    valid_metric = np.array([])
    times = np.array([])
    
    #For each model, find cv_score and metrics. Also store training time.
    for i,seed in enumerate(seeds):
        if verbose: progress.update("%d/%d: Seed = %d"%(i+1,N,seed))
        model = modmkr(seed,**model_maker_kwargs)
        cv_score = np.mean(cross_val_score(model,Xtra,ytra,scoring=metric,cv=cv))
        cv_scores = np.append(cv_scores,cv_score)
        
        start_time = time()
        model.fit(Xtra,ytra)
        train_metric = np.append(train_metric, metric(model,Xtra,ytra))
        if use_val: valid_metric = np.append(valid_metric, metric(model,Xval,yval))
        times = np.append(times, time()-start_time)

    #Plot a summary when done.
    if plot_summary:
        fig,ax = plt.subplots()
        scat = ax.scatter(train_metric,valid_metric,c=times,cmap='coolwarm',s=50)
        cbar = fig.colorbar(scat,ax=ax)
        ax.set_xlabel("Training Metric")
        ax.set_ylabel("Validation Metric")
        cbar.ax.set_ylabel("Training Time")
        mi,ma = 0,1.1*np.max([np.max(train_metric),np.max(valid_metric)])
        ax.set_xlim(mi,ma)
        ax.set_ylim(mi,ma)
        ax.plot([mi,ma],[mi,ma],ls='--',color='black')
    
    sort = np.argsort(cv_scores)
    return np.c_[seeds[sort],cv_scores[sort],train_metric[sort],valid_metric[sort],times[sort]]


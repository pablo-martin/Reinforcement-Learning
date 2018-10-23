'''
We will use a 'grid search' of alpha, beta parameter space for looking for
the best fit given a session
'''
import os
import time
import pickle
import pandas as pd
import numpy as np
from RescorlaWagner import grid_search

ROOT = os.environ['HOME'] + '/python/'
idx = pd.IndexSlice
noRuns = 100

target = ROOT + 'RLmodule/Results/RW_PSR.p'
DSR = pickle.load(open(ROOT + \
            'DATA_structures/session_dataframes/' + \
            'PSR_TRAINING_SESSIONS_DATAFRAME.p', 'rb'))
try:
    RWScores = pickle.load(open(target, 'rb'))
except IOError:
    new_index = DSR.index
    new_index = new_index.droplevel('trial')
    new_index = new_index.droplevel('block')
    new_index = new_index.drop_duplicates(keep='first')
    RWScores = pd.DataFrame(np.full((len(new_index),3), np.NaN),
                            index=new_index,
                            columns=['alpha','beta','score'])


for rat_label, rat_data in DSR.groupby('rat'):
    for session_label, session in rat_data.groupby('training_session'):
        if np.isnan(RWScores.loc[rat_label, session_label]).any():
            start = time.time()
            alpha, beta, score = grid_search(session, noRuns = noRuns)
            RWScores.loc[idx[rat_label, session_label],'alpha'] = alpha
            RWScores.loc[idx[rat_label, session_label],'beta'] = beta
            RWScores.loc[idx[rat_label, session_label],'score'] = score
            print('rat:%s - training-session:%s - time:%1.2f'\
                  %(rat_label, session_label, time.time() - start))

pickle.dump(RWScores, open(target, 'wb'))

'''
this will be a function that fits a temporal-difference Rescorla-Wagner
model to a given session

Inputs: session, alpha, beta, # of runs
Output: decoding accuracy
'''
from RatClass import Rat
import itertools
import numpy as np

def RWmodel(session, alpha = 0.03, beta = 0.1, noRuns = 10):
    if alpha <= 0: alpha = 0.03
    if beta <= 0: beta = 0.1
    if noRuns <= 0: noRuns = 10
    scores = []
    for run in range(noRuns):
        #create rat
        ratID = 'rat' + str(run)
        rat = Rat(ratID, alpha = alpha, beta = beta)
        #empty prediction vector
        preds = np.full(len(session), np.NaN)
        for trial in range(len(session)):
            preds[trial] = rat.make_decision()
            rat.update_beliefs(session['AR'].iloc[trial],
                               session['Choice'].iloc[trial])
        session.loc[:, ratID] = preds
        scores.append(np.float(np.sum(session.loc[:,'Choice'] ==
                                     session.loc[:,ratID])) / len(session))
    return scores

'''
This function searches a grid of the parameter space of alphas and betas, and
determines the best fit for a given session.
'''
def grid_search(session, noRuns = 10):
    alphas = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 1.5, 2, 5]
    betas = [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2, 5]
    scores = np.zeros((len(alphas), len(betas)), dtype=np.float)

    for alpha, beta in itertools.product(alphas, betas):
        grid_point = RWmodel(session,
                             alpha = alpha,
                             beta = beta,
                             noRuns = noRuns)
        scores[alphas.index(alpha), betas.index(beta)] = np.nanmean(grid_point)
    ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
    alpha = alphas[ind[0]]
    beta = betas[ind[1]]
    score = np.max(scores)
    return alpha, beta, score

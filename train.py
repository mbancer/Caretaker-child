import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import itertools, pdb, random, os
import numpy as np
from chatbots_MB import Team
import options
from time import gmtime, strftime
import matplotlib.pyplot as plt
from scipy.io import savemat
from analysis import *


def conditional_entropy(X, Y):
    """
    Returns conditional entropy H(X|Y)
    """    
    _, counts = np.unique(X, return_counts=True)
    probs_X = counts / counts.sum()
        
    _, counts = np.unique(Y, return_counts=True)
    probs_Y = counts / counts.sum()

    XY = np.vstack([X, Y]).T
    idx, counts = np.unique(XY, axis=0, return_counts=True)
    probs = counts / counts.sum()
    
    H_XY = -np.sum(probs*(np.log2(probs)))
    H_Y = -np.sum(probs_Y*(np.log2(probs_Y)))
    H_X = -np.sum(probs_X*(np.log2(probs_X)))

    return 1 - (H_XY - H_Y) / H_X

# read the command line options
options = options.read()
#------------------------------------------------------------------------
# setup experiment
#------------------------------------------------------------------------
params = {}
for key, value in options.items():
    params[key] = value

# parameters just to define embedding network sizes
params['mInVocab'] = params['cOutVocab'] + params['mOutVocab']
params['cInVocab'] = params['cOutVocab'] + params['mOutVocab'] + 9
#------------------------------------------------------------------------
# build agents, and setup optmizer
#------------------------------------------------------------------------
team = Team(params)
team.train()
optimizer = optim.AdamW([{'params': team.mBot.parameters(), \
                                'lr': params['learningRate']},\
                        {'params': team.cBot.parameters(), \
                                'lr': params['learningRate']}], weight_decay=0.05)

#------------------------------------------------------------------------
# train agents
#------------------------------------------------------------------------
# begin training
numIterPerEpoch = 1
count = 0

# keep track of some statistics
REW, stdREW = [], []
ma_cs, ma_ms, ms_cs, cs1_ms = [], [], [], []

for iterId in range(params['numEpochs'] * numIterPerEpoch):

    epoch = float(iterId)/numIterPerEpoch
    # forward pass
    talks = team.forward(params['batchSize'])
    # backward pass
    batchReward = team.backward(optimizer, epoch)
    REW.append(team.totalReward.cpu().item())
    stdREW.append(team.stdReward.cpu().item())
    # take a step by optimizer
    optimizer.step()

    #if iterId >= 0 and iterId % (10000*numIterPerEpoch) == 0:
    #    team.saveModel(savePath, optimizer, params)

    if iterId % 100 != 0:
        continue

    cs = np.array([talks[ii][1].cpu().numpy() for ii in range(len(talks))])
    ms = np.array([talks[ii][2].cpu().numpy() for ii in range(len(talks))])
    ma = np.array([talks[ii][3].cpu().numpy() for ii in range(len(talks))])

    ma_cs.append(conditional_entropy(ma.flatten(), cs.flatten()))
    ma_ms.append(conditional_entropy(ma.flatten(), ms.flatten()))
    ms_cs.append(conditional_entropy(ms.flatten(), cs.flatten()))
    cs1_ms.append(conditional_entropy(cs[1:].flatten(), ms[:-1].flatten()))

    time = strftime("%a, %d %b %Y %X", gmtime())
    print('[%s][Iter: %d][Ep: %.2f][R: %.4f][AvgSt: %.4f]' % \
          (time, iterId, epoch, team.totalReward, team.childState.cpu().mean()))
    #------------------------------------------------------------------------   


# save model and learning curves to a corresponding dir saveDir
saveDir = 'models_' + str(params['numRounds']) + '/' + team.decay + '_decay/N_' + str(params['numVitalParams']) + '_vc_' + str(params['cOutVocab']) + '_vm_' + str(params['mOutVocab']) + "_lr_" + str(params['learningRate']) + '_hc_' + str(params['cHiddenSize']) + '_hm_' + str(params['mHiddenSize']) + '_cSpeakFreq_' + str(params['cSpeakFreq']) + '_mActFreq_' + str(params['mActFreq']) + '/'
print('Saving to path: ' + saveDir)

isExist = os.path.exists(saveDir)
if not isExist:
    os.makedirs(saveDir)

ix = options['ix']

fig, ax = plt.subplots(1, 1, figsize = [6,6])
ax.plot(REW)
ax.set_title('Total reward, $N_{vital} = $' + str(team.numParams) + ', $v_{child}$ = ' + str(options['cOutVocab']) + ', $v_{mother}$ = ' + str(options['mOutVocab']))
plt.savefig(saveDir + 'learning_curve_par_' + str(team.numParams) + '_voc_' + str(options['cOutVocab']) + '_' + str(ix) + '.png')

team.saveModel(saveDir + 'model_' + str(ix), optimizer, params)
savemat(saveDir + 'learning_curve_'+ str(ix) + '.mat', {'totalReward': REW, 'stdReward': stdREW, 'ma_cs': ma_cs, 'ma_ms': ma_ms, 'ms_cs': ms_cs, 'cs1_ms': cs1_ms})

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import itertools, pdb, random, json
import numpy as np
from chatbots_MB import Team

import sys
sys.path.append('../')
#from utilities import saveResultPage
from scipy.io import savemat, loadmat
import pickle
from analysis import draw_talk, draw_states

#------------------------------------------------------------------------
# load experiment and model
#------------------------------------------------------------------------
if len(sys.argv) < 2:
    print('Wrong usage:')
    print('python test.py <modelPath>')
    sys.exit(0)

# load the model
loadPath = sys.argv[1]
loaded = torch.load(loadPath)
savePath = loadPath.split('/')[:-1]
savePath = '/'.join(savePath) 
params = loaded['params']
team = Team(params)
team.loadModel(loaded)

print('Loaded model from: %s' % loadPath)

#------------------------------------------------------------------------
# produce talk & reward
#------------------------------------------------------------------------
#team.evaluate()

talks = team.forward(params['batchSize'])

ctr = 0
ct_cumul, mt_cumul, ma_cumul, st_cumul = [], [], [], np.zeros((1, params['numVitalParams']))
for jj in range(params['batchSize']):
    if jj % 100 == 0:
        print(str(100 * jj/params['batchSize']) + '%')

    ctr += 1
    #talk = [talks[ix][jj] for ix in range(len(talks))]
    #talk = [int(t) for t in talk]

    REW_tmp = [float(team.get_reward(talks[i][0])[jj]) for i in range(params['numRounds'])]
    childState = np.array([talks[i][0][jj].cpu().numpy() for i in range(params['numRounds'])])
    childTalk = [talks[i][1][jj].cpu().numpy() for i in range(params['numRounds'])]
    motherTalk = [talks[i][2][jj].cpu().numpy() for i in range(params['numRounds'])]
    motherActs = [talks[i][3][jj].cpu().numpy() for i in range(params['numRounds'])]

    ct_cumul.append(childTalk)
    mt_cumul.append(motherTalk)
    ma_cumul.append(motherActs)

    st_cumul = np.concatenate((st_cumul, childState), axis=0)

#    if jj % 100 == 0: 
#        draw_talk(childTalk, motherTalk, motherActs, savePath + '/talk_' + str(jj))
#        draw_states(childState.T, REW_tmp, team.opt, team.A, team.tau, savePath + '/states_' + str(jj))

ct_cumul = [item for sublist in ct_cumul for item in sublist]
mt_cumul = [item for sublist in mt_cumul for item in sublist]
ma_cumul = [item for sublist in ma_cumul for item in sublist]
st_cumul = st_cumul[1:, :]

savemat(savePath + '/data_cumul.mat', {'states': st_cumul, 'ct': ct_cumul, 'mt': mt_cumul, 'ma': ma_cumul})

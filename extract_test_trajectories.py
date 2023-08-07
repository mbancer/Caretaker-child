import sys
import pickle
import glob
import torch
import numpy as np
from scipy.io import loadmat

from chatbots_MB import Team

#------------------------------------------------------------------------
# load experiment and model
#------------------------------------------------------------------------
if len(sys.argv) < 2:
    print('Usage:')
    print('python test.py <loadPath>')
    sys.exit(0)


childStates, childSignals, motherSignals, motherActions = [], [], [], []

lpath = sys.argv[1]

for f in glob.glob(f"{lpath}/model_*"):
    loaded = torch.load(f)
    params = loaded['params']
    team = Team(params)
    team.loadModel(loaded)

    print(f"Loaded model from: {f}")

    #------------------------------------------------------------------------
    # produce talk & reward
    #------------------------------------------------------------------------
    #team.evaluate()
    talks = team.forward(params['batchSize'])

    ct_cumul, mt_cumul, ma_cumul, st_cumul = [], [], [], np.zeros((1, params['numVitalParams']))

    childStates.append(np.array([talks[i][0].cpu().numpy() for i in range(params['numRounds'])]))
    childSignals.append(np.array([talks[i][1].cpu().numpy() for i in range(params['numRounds'])]))
    motherSignals.append(np.array([talks[i][2].cpu().numpy() for i in range(params['numRounds'])]))
    motherActions.append(np.array([talks[i][3].cpu().numpy() for i in range(params['numRounds'])]))

childStates = np.stack(childStates)
childSignals = np.stack(childSignals)
motherSignals = np.stack(motherSignals)
motherActions = np.stack(motherActions)

data = {'childStates': childStates, 'childSignals': childSignals,
        'motherSignals': motherSignals, 'motherActions': motherActions}
with open(f"{lpath}/test_trajectories.pkl", "wb") as f:
    pickle.dump(data, f)

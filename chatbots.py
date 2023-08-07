# class defintions for chatbots - questioner and answerer

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import sys
from utilities import initializeWeights
import numpy as np
from random import uniform, randrange, sample

#---------------------------------------------------------------------------
# Parent class for both bots
class ChatBot(nn.Module):
    def __init__(self, params):
        super(ChatBot, self).__init__()

        # absorb all parameters to self
        for attr in params: setattr(self, attr, params[attr])

        # standard initializations
        self.hState = torch.Tensor()
        self.cState = torch.Tensor()
        self.actions = []
        self.messages = []
        self.evalFlag = False

        # modules (common)
        self.inNet = nn.Embedding(self.inVocabSize, self.embedSize)
        self.outNet = nn.Linear(self.hiddenSize, self.outVocabSize + self.numActions)

        # initialize weights
        initializeWeights([self.inNet, self.outNet], 'xavier')

    # initialize hidden states
    def resetStates(self, batchSize, retainActions=False):
        # create tensors
        self.hState = torch.Tensor(batchSize, self.hiddenSize)
        self.hState.fill_(0.0)
        self.hState = Variable(self.hState)
        self.cState = torch.Tensor(batchSize, self.hiddenSize)
        self.cState.fill_(0.0)
        self.cState = Variable(self.cState)

        if self.useGPU:
            self.hState = self.hState.cuda()
            self.cState = self.cState.cuda()

        # new episode
        if not retainActions:
            self.messages = []
            self.actions = []

    # freeze agent
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    # unfreeze agent
    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    # given an input token, interact for the next round
    def listen(self, inputToken, imgEmbed=None):
        # embed and pass through LSTM
        tokenEmbeds = self.inNet(inputToken)
        # concat with image representation
        if imgEmbed is not None:
            tokenEmbeds = torch.cat((tokenEmbeds, imgEmbed), 1)

        # now pass it through rnn
        self.hState, self.cState = self.rnn(tokenEmbeds,
                                            (self.hState, self.cState))

    # speak a token
    def speak(self):
        # compute softmax and choose a token
        outMessages = self.outNet(self.hState)[:, :self.outVocabSize]
        outActions = self.outNet(self.hState)[:, self.outVocabSize:]

        #outDistr = nn.functional.softmax(self.outNet(self.hState), dim=-1)
        outActionsDistr = nn.functional.softmax(outActions, dim=-1)
        outMessagesDistr = nn.functional.softmax(outMessages, dim=-1)
        #print('outDistr: ',outDistr[:10])
        # if evaluating
        if self.evalFlag:
            _, actions = outActionsDistr.max(1)
            _, messages = outMessagesDistr.max(1)
        else:
            action_sampler = torch.distributions.Categorical(outActionsDistr)
            message_sampler = torch.distributions.Categorical(outMessagesDistr)
            #print('distr.cat: ', action_sampler)
            actions = action_sampler.sample()
            messages = message_sampler.sample()
            #print('actions: ',actions[:10],min(actions),max(actions))
            #print('messages: ',messages[:10],min(messages),max(messages))
            # record actions
            self.actions.append(-action_sampler.log_prob(actions))
            self.messages.append(-message_sampler.log_prob(messages))

        if self.useGPU:
            actions = actions.cuda()
            messages = messages.cuda()

        return actions, messages 

    # reinforce each state with reward
    def reinforce(self, rewards):
        #self.actions = self.actions * rewards
        #print('len act: ' + str(len(self.actions)))
        for index, action in enumerate(self.actions):
           self.actions[index] = action * rewards[:,index]
        for index, message in enumerate(self.messages):
           self.messages[index] = message * rewards[:,index]

    # backward computation
    def performBackward(self):
        sum([ii.sum() for ii in self.actions + self.messages]).backward()

    # switch mode to evaluate
    def evaluate(self):
        self.evalFlag = True

    # switch mode to train
    def train(self):
        self.evalFlag = False


#---------------------------------------------------------------------------
class Mother(ChatBot):
    def __init__(self, params):
        self.parent = super(Mother, self)
        # input-output for current bot
        params['inVocabSize'] = params['mInVocab']
        params['outVocabSize'] = params['mOutVocab']
        params['hiddenSize'] = params['mHiddenSize']
        self.numActions = params['numVitalParams'] + 1
        try: self.actFreq = params['mActFreq']
        except: self.actFreq = 1
        self.parent.__init__(params)

        # rnn inputSize
        #self.mInsightSize = params['mInsightSize']

        self.rnn = nn.LSTMCell(self.embedSize, self.hiddenSize)
        initializeWeights([self.rnn], 'xavier')#, self.imgNet], 'xavier')

        # set offset
        self.listenOffset = params['cOutVocab']

#---------------------------------------------------------------------------
class Child(ChatBot):
    def __init__(self, params):
        self.parent = super(Child, self)
        # input-output for current bot
        params['inVocabSize'] = params['cInVocab']
        params['outVocabSize'] = params['cOutVocab']
        params['hiddenSize'] = params['cHiddenSize']
        self.numActions = 1
        self.speakFreq = params['cSpeakFreq']
        self.parent.__init__(params)

        self.rnn = nn.LSTMCell(self.embedSize + params['numVitalParams'], self.hiddenSize)

	# set offset
        self.listenOffset = params['mOutVocab']



#---------------------------------------------------------------------------
class Team:
    # initialize
    def __init__(self, params):
        # memorize params
        for field, value in params.items():
            setattr(self, field, value)
        self.mBot = Mother(params)
        self.cBot = Child(params)
        self.criterion = nn.NLLLoss()
        self.numParams = params['numVitalParams']

        self.reward = torch.zeros(params['batchSize'], (self.numRounds + 1))
        self.totalReward = None

        self.decay = params['decay']
        if self.decay == 'log':
            self.A = torch.linspace(-1, -1, self.numParams)
            self.tau = torch.linspace(3.5, 5.5, self.numParams)
            self.opt = -4
        elif self.decay == 'exp':
            self.A = torch.linspace(.7, .7, self.numParams)
            self.tau = torch.linspace(1.5, 3, self.numParams)
            self.opt = 0.5

        self.batchSize = params['batchSize']

        self.initialize_child()
        self.initialReply = torch.zeros(self.batchSize).int() + self.numParams

        # ship to gpu if needed
        if self.useGPU:
            self.mBot = self.mBot.cuda()
            self.cBot = self.cBot.cuda()
            self.reward = self.reward.cuda()
            #self.initialState = self.initialState.cuda()
            self.initialReply = self.initialReply.cuda()
            self.A = self.A.cuda()
            self.tau = self.tau.cuda()

        print(self.mBot)
        print(self.cBot)


    # update child's state
    def update_child(self, states, actions):

        states_new = states.clone().detach()

        if self.decay == 'log':
            states_new = -torch.abs(states) * torch.log(self.tau)
        elif self.decay == 'exp':
            states_new = states * torch.exp(-1/self.tau) 
        
        for ii in range(self.numParams):  
            states_new[torch.where(actions==ii)[0], ii] = self.A[ii]

        #states_new = torch.clamp(states_new, min = -20)
        return states_new 
	
    	
    def initialize_child(self):
        if self.decay == 'log': 
            #max_decay = -torch.log(self.tau.cpu())**self.numRounds
            max_decay = -200
            self.childState = torch.rand((self.batchSize, self.numParams)) * max_decay
        else:
            self.childState = torch.full((self.batchSize, self.numParams), self.A[0]) #or opt 
        
        if self.useGPU:
            self.childState = self.childState.cuda()

    # get reward
    def get_reward(self, state):
        #print('state shape=', state.shape)
        #avg_state = state.sum(1)
        #print(((state - self.opt)**2).sum(axis=1).size())
        return -((state - self.opt)**2).sum(axis=1) #/ 0.01 + 10

    # switch to train
    def train(self):
        self.mBot.train()
        self.cBot.train()

    # switch to evaluate
    def evaluate(self):
        self.mBot.evaluate()
        self.cBot.evaluate()

    # forward pass
    def forward(self, batchSize, record = False):
        # reset the states of the bots
        self.cBot.resetStates(batchSize)
        self.mBot.resetStates(batchSize)

        mBotReply = self.initialReply
        # if the conversation is to be recorded
        talk = []
        self.initialize_child()
        self.states = []

        for ii, roundId in enumerate(range(self.numRounds)):
            # listen to answer, ask q_r, and listen to q_r as well
            self.cBot.listen(mBotReply, self.childState) 
            
            if ii % self.cBot.speakFreq == 0: 
                _, cBotQues = self.cBot.speak()
                # clone
                cBotQues = cBotQues.detach()
                self.cBot.listen(self.cBot.listenOffset + cBotQues, self.childState)

                # listen to question and answer, also listen to answer
                self.mBot.listen(cBotQues)
                
            if ii % self.mBot.actFreq == 0:
                acts, mBotReply = self.mBot.speak()
                mBotReply = mBotReply.detach()
                self.mBot.listen(mBotReply + self.mBot.listenOffset)
            
            self.states.append(self.childState)
            talk.append([self.childState, cBotQues, mBotReply, acts])
            self.childState = self.update_child(self.childState, acts)
            

        # listen to the last answer
        self.cBot.listen(mBotReply, self.childState)
        return talk

    # backward pass
    def backward(self, optimizer, epoch, baseline = None):
        # compute reward

        # REINFORCE REWARD MB
        gamma = .5
        for jj in range(self.numRounds, -1, -1):
            if jj == self.numRounds:
                self.reward[:, jj] = self.get_reward(self.childState)
            else:
                self.reward[:, jj] = self.get_reward(self.states[jj]) + gamma * self.reward[:, jj + 1]

        # reinforce all actions for cBot, mBot
        self.cBot.reinforce(self.reward)
        self.mBot.reinforce(self.reward)

        # optimize
        optimizer.zero_grad()
        self.cBot.performBackward()
        self.mBot.performBackward()

        # clamp the gradients
        for p in self.cBot.parameters():
          p.grad.data.clamp_(min=-5., max=5.)
        for p in self.mBot.parameters():
          p.grad.data.clamp_(min=-5., max=5.)

        # cummulative reward
        batchReward = self.reward/self.rlScale
        if self.totalReward == None: self.totalReward = batchReward
        self.stdReward = torch.std(0.95 * self.totalReward + 0.05 * batchReward)
        self.totalReward = torch.mean(0.95 * self.totalReward + 0.05 * batchReward)
        return torch.mean(batchReward)

    # loading modules from saved model
    def loadModel(self, savedModel):
        modules = ['rnn', 'inNet', 'outNet', 'imgNet',
                   'predictRNN', 'predictNet']
        # savedModel is an instance of dict
        dictSaved = isinstance(savedModel['cBot'], dict)

        for agentName in ['mBot', 'cBot']:
            agent = getattr(self, agentName)
            for module in modules:
                if hasattr(agent, module):
                    if dictSaved: savedModule = savedModel[agentName][module]
                    else: savedModule = getattr(savedModel[agentName], module)
                    # assign to current model
                    setattr(agent, module, savedModule)


    # saving module, at given path with params and optimizer
    def saveModel(self, savePath, optimizer, params):
        modules = ['rnn', 'inNet', 'outNet', 'imgNet',
                   'predictRNN', 'predictNet']
        toSave = {'mBot': {}, 'cBot': {}, 'params': params, 'optims': optimizer}
        for agentName in ['mBot', 'cBot']:
            agent = getattr(self, agentName)
            for module in modules:
                if hasattr(agent, module):
                    toSaveModule = getattr(agent, module)
                    toSave[agentName][module] = toSaveModule
        # save checkpoint.
        torch.save(toSave, savePath)

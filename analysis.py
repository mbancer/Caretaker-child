import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score

def draw_states(states, rew_tmp, opt, A, tau, fname):
	fig, ax = plt.subplots()
	c = np.zeros([states.shape[0], 3])
	c[:, 0] = np.linspace(1, 0, states.shape[0])
	c[:, -1] = np.linspace(0, 1, states.shape[0])
	for ii in range(states.shape[0]):
		ax.plot(np.arange(1, 13), states[ii, :], label = '$P_' + str(ii) + '$' +  r', $\tau_' + str(ii) +  '$ : ' + str(tau[ii]), color = c[ii])
	ax.plot([0, states.shape[1]], [opt, opt], ':k')
	ax2 = ax.twinx()
	ax2.plot(np.arange(1, 13), rew_tmp, '--k')
	ax.set_ylabel('$child$ state ($P$)')
	ax.legend()
	ax.set_xlim([1, 12])
	ax2.set_ylabel('immediate reward (--)')
	ax.set_xlabel('rounds')
	plt.savefig(fname + '.png', bbox_inches = 'tight')  
	plt.close()
	
def draw_talk(child_talk, mother_talk, mother_acts, fname):
	fig,ax=plt.subplots()
	M = np.array([child_talk, mother_talk, mother_acts])
	c=ax.imshow(M, cmap = 'cool')
	ax.set_xlabel('rounds')
	ax.set_yticks([0, 1, 2])
	ax.set_xticks(np.arange(1, 12, 2))
	ax.set_xticklabels(np.arange(2, 13, 2).tolist())
	ax.set_yticklabels(['$m_c$', '$m_m$', '$a_m$'])
	[ax.text(x-.1, 0.1, child_talk[x], color = 'black') for x in range(len(child_talk))]
	[ax.text(x-.1, 1.1, mother_talk[x], color = 'black') for x in range(len(child_talk))]
	[ax.text(x-.1, 2.1, mother_acts[x], color = 'black') for x in range(len(child_talk))]
	plt.colorbar(c, location = 'top')
	#plt.title('$N = 4, v_{c} = v_{m} = 3$')
	#plt.text(4,-2.5,'$N = 6, v_{c} = v_{m} = 1$')
	plt.savefig(fname+'.png', bbox_inches = 'tight')
	plt.close()
         
def primitive_RQA(series_1, series_2, fname, s):
	M = np.zeros([len(series_1), len(series_2)])
	for ii in range(M.shape[0]):
		for jj in range(M.shape[1]):
			if series_1[ii] == series_2[jj]: M[ii,jj] = 1 
	fig, ax = plt.subplots()
	ax.imshow(M, cmap = 'binary')
	ax.xaxis.tick_top()
	ax.set_yticks([x for x in range(len(series_1))])
	ax.set_yticklabels([str(int(x)) for x in series_1])
	ax.set_xticks([x for x in range(len(series_2))])
	ax.set_xticklabels([str(int(x)) for x in series_2])
	if s == 'ct-mt':
		ax.set_ylabel('$child$ messages, $m_c$')
		ax.set_xlabel('$mother$ messages, $m_m$')
	elif s == 'ct-ma':
		ax.set_ylabel('$child$ messages, $m_c$')
		ax.set_xlabel('$mother$ actions, $a_m$')
	elif s == 'mt-ma':
		ax.set_ylabel('$mother$ messages, $m_m$')
		ax.set_xlabel('$mother$ actions, $a_m$')
	ax.xaxis.set_label_position('top')
	plt.savefig(fname + '_' + s + '.png') 
	plt.close()
	#plt.show()
	return M
	
def primitive_correlation(x, y):
	r, pval1 = scipy.stats.pearsonr(x, y);
	rho, pval2 = scipy.stats.spearmanr(x, y);
	tau, pval3 = scipy.stats.kendalltau(x, y);
	#print('Pearson: ')
	#print(r, pval1)
	#print('Spearman: ')
	#print(rho, pval2)
	#print('Kendall: ')
	#print(tau, pval3)
	return [(r, pval1), (rho, pval2), (tau, pval3)]
	
def save_correlations(corrs, fname):
	with open(fname + '.txt', 'w') as f:
		f.write('Pearson, Spearman, Kendall (and corresponding p-vals)\n')
		for ii in range(len(corrs)):
			for jj in range(3):
		    		f.write(format(corrs[ii][jj][0], '.4f')+" ("+format(corrs[ii][jj][1],'.4f')+") , ")
			f.write("\n")
	fig,ax = plt.subplots()
	plt.hist([corrs[i][0][0] for i in range(len(corrs))], alpha = .5, label = 'pearson')
	plt.hist([corrs[i][1][0] for i in range(len(corrs))], alpha = .5, label = 'spearman')
	plt.hist([corrs[i][2][0] for i in range(len(corrs))], alpha = .5, label = 'kendall')
	plt.legend()
	plt.xlabel('correlation')
	plt.savefig(fname+'_hist.png')
	plt.close()
			
def mutual_information(X, Y, bins):
	p_XY = np.histogram2d(X, Y, [bins[0], bins[1]])[0]
	return mutual_info_score(None, None, contingency = p_XY) # expressed in nats

def mi_JZ(X, Y):
	_, counts = np.unique(X, return_counts=True)
	probs_X = counts / counts.sum()
    
	_, counts = np.unique(Y, return_counts=True)
	probs_Y = counts / counts.sum()

	XY = np.vstack([X, Y]).T
	_, counts = np.unique(XY, axis=0, return_counts=True)
	probs = counts / counts.sum()

	H_X = scipy.stats.entropy(probs_X, base=2)
	H_Y = scipy.stats.entropy(probs_Y, base=2)
	H_XY = scipy.stats.entropy(probs, base=2)

	I_XY = H_X + H_Y - H_XY

	return 2 * I_XY / (H_X + H_Y)

def entropy_JZ(X):
	_, counts = np.unique(X, return_counts=True)
	probs_X = counts / counts.sum()
	H_X = scipy.stats.entropy(probs_X, base=2)
	return H_X

def entropy(X, bins):
	#bins = X.shape[0]
	p_X = np.histogram(X, bins)[0]
	p_X = p_X/np.sum(p_X)
	#print(p_X)
	return scipy.stats.entropy(p_X)



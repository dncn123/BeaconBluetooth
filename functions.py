import pandas as pd
import numpy as np
import pymc as pm

# http://math.stackexchange.com/a/1367732
def TwoCirclesInters(c1,r1,c2,r2):
	x1,y1 = c1
	x2,y2 = c2
	dx = x1 - x2
	dy = y1 - y2
	R = np.sqrt(np.power(dx,2) + np.power(dy,2))
	if (R > r1 + r2):
		return np.array([[-1,-1],[-1,-1]])

	Rsq = np.power(R,2)
	Rcb = np.power(R,4)

	r2r2 = (np.power(r1,2) - np.power(r2,2))
	a = r2r2 / (2*Rsq)

	b = np.sqrt(2 * (np.power(r1,2) + np.power(r2,2)) / Rsq
				- np.power(r2r2,2) / Rcb - 1)

	cx1 = 0.5*(x1+x2) + a*(x2-x1) + 0.5*b*(y2-y1)
	cx2 = 0.5*(x1+x2) + a*(x2-x1) - 0.5*b*(y2-y1)

	cy1 = 0.5*(y1+y2) + a*(y2-y1) + 0.5*b*(x1-x2)
	cy2 = 0.5*(y1+y2) + a*(y2-y1) - 0.5*b*(x1-x2)

	return np.array([[cx1, cy1], [cx2, cy2]])

def bayes_lin_regr(signal, distance, iter, burn, thin):
	beta_prior_mean = np.mean(distance/signal)

	alpha = pm.Normal('alpha', 0, 0.001)
	beta = pm.Normal('beta', beta_prior_mean, 0.001)

	signal = pm.Normal('signal', mu=0, tau=0.5, 
						value=signal, 
						observed=True)

	@pm.deterministic()
	def linear_regress(signal=signal, alpha=alpha, beta=beta):
	    return signal*beta+alpha

	error = pm.Uniform('error', 0, 500)

	distance = pm.Normal('distance', mu=linear_regress, tau=error,
					value=distance, observed=True)

	model = pm.Model([signal, distance,
						alpha, beta, error])

	mcmc = pm.MCMC(model)
	mcmc.sample(iter=iter, burn=burn, thin=thin)
	return mcmc

def CircleCrosses(A,B,C,disA,disB,disC):

	a=TwoCirclesInters(A,disA[0],B,disB[0])
	b=TwoCirclesInters(A,disA[0],C,disC[0])
	c=TwoCirclesInters(B,disB[0],C,disC[0])
	crosses = np.concatenate((a,b,c))

	for i in range(1,len(disA)):
		a=TwoCirclesInters(A,disA[i],B,disB[i])
		b=TwoCirclesInters(A,disA[i],C,disC[i])
		c=TwoCirclesInters(B,disB[i],C,disC[i])
		crosses = np.concatenate((crosses,a,b,c), axis=0)

	subset = (crosses[:,0]>=0) & (crosses[:,1]>0) &\
				(crosses[:,0]<=274) & (crosses[:,1]<=454)

	return crosses[subset]


























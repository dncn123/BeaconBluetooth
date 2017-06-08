import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from pymc.Matplot import plot as mcplot
from functions import TwoCirclesInters, bayes_lin_regr, CircleCrosses
data = pd.read_csv("beacon_readings.csv")
del data["Date"]
del data["Time"]

data.head()

data.columns = ['Signal_A', 'Signal_B',
				'Signal_C', 'Position_X',
       			'Position_Y']

# Beacon Centres
A = (10, 180)
B = (122, 454)
C = (274, 266)

data.Position_Y.unique()

# How how signal strength compare to distance
data['Distance_A'] = np.sqrt(np.power(data.Position_X-A[0],2) +
							np.power(data.Position_Y-A[1],2))

data['Distance_B'] = np.sqrt(np.power(data.Position_X-B[0],2) +
							np.power(data.Position_Y-B[1],2))

data['Distance_C'] = np.sqrt(np.power(data.Position_X-C[0],2) +
							np.power(data.Position_Y-C[1],2))

data['Distance_A'].unique()

subset = (data['Signal_A']==0)|\
		 (data['Signal_B']==0)|\
		 (data['Signal_C']==0)

data = data[~subset]

# Relationship between signal and distance
plt.scatter(data['Signal_A'], data['Distance_A'], label='A')
plt.scatter(data['Signal_B'], data['Distance_B'], label='B')
plt.scatter(data['Signal_C'], data['Distance_C'], label='C')
plt.legend()
plt.savefig("singal_to_distance.png")
plt.clf()

# Each detector looks like it has it's own calibration 
# so will model them separately

# Bayesian Regression to convert signal to distance

mcmcA = bayes_lin_regr(data['Signal_A'].values, 
						data['Distance_A'].values,
						200000, 10000, 50)

mcmcB = bayes_lin_regr(data['Signal_B'].values, 
						data['Distance_B'].values,
						200000, 10000, 50)

mcmcC = bayes_lin_regr(data['Signal_C'].values, 
						data['Distance_C'].values,
						200000, 10000, 50)

# Spit out mcmc statistics

names = ["mcmcA", "mcmcB", "mcmcC"]
for i, mcmc in enumerate([mcmcA, mcmcB, mcmcC]):
	trc = getattr(mcmc, "trace")
	mcplot(trc('alpha'))
	plt.savefig(names[i]+"_alpha.png")
	mcplot(trc('beta'))
	plt.savefig(names[i]+"_beta.png")
	mcplot(trc('error'))
	plt.savefig(names[i]+"_error.png")

# Now we need a function that can take the 3 circles
# and estimate where they all cross

data.iloc[1]

Signal_A_=1.201608
Signal_B_=1.031228
Signal_C_=1.893498
Position_X_=122
Position_Y_=180
Distance_A_=112
Distance_B_=274
Distance_C_=174.642492

# using the traces

RsA = Signal_A_*mcmcA.trace('beta')[:3000]+\
				mcmcA.trace('alpha')[:3000]+\
				pm.rnormal(0,mcmcA.trace('error')[:3000])

RsB = Signal_B_*mcmcB.trace('beta')[:3000]+\
				mcmcB.trace('alpha')[:3000]+\
				pm.rnormal(0,mcmcB.trace('error')[:3000])

RsC = Signal_C_*mcmcC.trace('beta')[:3000]+\
				mcmcC.trace('alpha')[:3000]+\
				pm.rnormal(0,mcmcC.trace('error')[:3000])

points = CircleCrosses(A,B,C,RsA,RsB,RsC)

xA, yA = A
xB, yB = B
xC, yC = C

ax = plt.gca()
ax.cla()
ax.set_aspect('equal')
ax.set_ylim((0,454))
ax.set_xlim((0,274))
ax.scatter([xA,xB,xC], [yA,yB,yC], color='g', label="Senor")
ax.scatter(points[:,0], points[:,1], 
			color='black', alpha=0.002)
for r in RsA:
	circle = plt.Circle(A, r, color='b', 
						fill=False, alpha=0.002)
	ax.add_artist(circle)

for r in RsB:
	circle = plt.Circle(B, r, color='b', 
						fill=False, alpha=0.002)
	ax.add_artist(circle)

for r in RsC:
	circle = plt.Circle(C, r, color='b', 
						fill=False, alpha=0.002)
	ax.add_artist(circle)

ax.scatter(Position_X_, Position_Y_, 
			color='r', label="true_position")
plt.legend()
plt.savefig("point_1_posterior.png")

































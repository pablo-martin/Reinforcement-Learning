import numpy as np
import pandas as pd

class SessionSimulator(object):

	def __init__(self,
				 Rat,
				 sessionID,
				 task = 'DSR',
				 noTrials = 120):

		if task == 'DSR':
			self.p = 1
		elif task == 'PSR':
			self.p = 0.85
		self.task = task
		self.noTrials = noTrials
		self.info = pd.DataFrame(np.zeros((self.noTrials, 6)),
								 columns = ['GA','Choice','Correct','AR','Q_W','Q_E'])
		self.Rat = Rat
		self.sessionID = sessionID

	def run_session(self):
		#block
		block = [0] * 12
		#randomly select the initial goal
		GA = int(np.random.random() > 0.5)

		for trial in range(self.noTrials):
			#record current goal arm
			self.info['GA'].iat[trial] = GA

			#trial starts with rat making a choice
			choice = self.Rat.make_decision()

			#saving that choice
			self.info['Choice'].iat[trial] = choice

			#see if that's the correct choice
			correct = int(choice == GA)
			#save that information
			self.info['Correct'].iat[trial] = correct

			#determine whether you actually get a reward
			if correct == 1:
				reward = int(np.random.random() <= self.p)
			elif correct == 0:
				reward = int(np.random.random() <= (1 - self.p))
			#save that information
			self.info['AR'].iat[trial] = reward
			self.info['Q_W'].iat[trial] = rat.Q[0]
			self.info['Q_E'].iat[trial] = rat.Q[1]

			#update your belief about the environment
			self.Rat.update_beliefs(reward, choice)



			#see if a reversal has been accomplished
			#update block
			block[0:-1] = block[1:]
			block[-1] = correct
			#check if criteria happened
			if np.sum(block) >= 10:
				#reset counter
				block = [0] * 12
				#reversal
				GA = (GA + 1) % 2
		rev_points = np.nonzero(self.info['GA'].diff())[0]
		rev_points = [w for w in rev_points] + [self.noTrials]
		rev_points = np.diff(rev_points)
		mtuples = [(self.Rat.ID, self.sessionID, i + 1, v) \
					   for i,w in enumerate(rev_points) for v in range(w)]
		index = pd.MultiIndex.from_tuples(mtuples, names=['rat','training_session','block','trials'])
		self.info.index = index


if __name__ == '__main__':
	#for debugging
	# rat = Rat(ratID = '22', alpha = 0.03, beta = 0.1)
	# SS = SessionSimulator(rat, 1, task = 'PSR', noTrials = 120)
	# SS.run_session()

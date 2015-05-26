import csv
import numpy as np

print 'Loading data...'

with open('data.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	x = list(reader)
	
	rawdata = np.array(x).astype('float')
	
	print 'Done'
	
	X = rawdata[np.all(rawdata != 0, axis=1)]
	
	m = np.mean(X,axis=0)
	
	Xshifted = X - m
	
	print 'Computing SVD...'
	
	U, S, V = np.linalg.svd(Xshifted, full_matrices=True)
	
	print 'done'
	#print U.shape, V.shape, S.shape
	
	d = 77
	
	Q = 0.5*np.eye(d)
	
	print 'Forward Pass'
	state = []
	state_pred = []
	cov_pred = []
	cov = []
	cov.insert(0,10000*np.eye(d))
	state.insert(0,np.random.normal(0.0,1.0,d))
	cov_pred.insert(0,10000*np.eye(d))
	state_pred.insert(0,np.random.normal(0.0,1.0,d))
	for i in range(0,1000):#range(0,rawdata.shape[0]):
		
		z =  rawdata[i,(rawdata[i,:]!=0)]
		H = np.diag(rawdata[i,:]!=0)
		H = H[~np.all(H == 0, axis=1)]
		Ht = np.dot(H,V[:,0:d])
		
		R = 1e-3*np.eye(H.shape[0])
				
		state_pred.insert(i,state[i])
		cov_pred.insert(i,cov[i] + Q)
		
		K = np.dot(np.dot(cov_pred[i],Ht.T),np.linalg.inv(np.dot(np.dot(Ht,cov_pred[i]),Ht.T) + R))
		
		state.insert(i,state_pred[i] + np.dot(K,(z - (np.dot(Ht,state_pred[i])+np.dot(H,m)))))
		cov.insert(i,np.dot(np.eye(d) - np.dot(K,Ht),cov_pred[i]))
		
	print 'Backward Pass'
	y = np.zeros((1000,rawdata.shape[1]))
	for i in range(1000,1,-1):
		state[i] =  state[i] + np.dot(np.dot(cov[i],np.linalg.inv(cov_pred[i])),(state[i+1] - state_pred[i+1]))
		cov[i] =  cov[i] + np.dot(np.dot(np.dot(cov[i],np.linalg.inv(cov_pred[i])),(cov[i+1] - cov_pred[i+1])),cov[i])
		
		y[i-1,:] = np.dot(V[:,0:d],state[i]) + m
		
	np.savetxt("out.csv", y, delimiter=",")

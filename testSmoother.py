import csv
import numpy as np
import smooth

print 'Loading data...'

with open('data.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	x = list(reader)
	
	rawdata = np.array(x).astype('float')
	
	print 'Done'
	
	y = smooth.smooth(rawdata,0.0025,1e-3)
	
	print 'Saving output'
	np.savetxt("out.csv", y, delimiter=",")

import csv
import numpy as np
import smooth
import time

print 'Loading data from data.csv'

with open('data.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	x = list(reader)
	
	rawdata = np.array(x).astype('float')
	
	print 'Done'
	
	start_time = time.time()
	y = smooth.smooth(rawdata,0.0025,1e-3)
	print ("%s seconds to process" % (time.time() - start_time))
	
	print 'Saving output to out.csv'
	np.savetxt("out.csv", y, delimiter=",")

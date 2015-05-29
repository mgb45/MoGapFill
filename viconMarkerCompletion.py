import sys

sys.path.append("C:\Program Files\Vicon\Nexus2.1\SDK\Python")

import ViconNexus
import numpy as np
import smooth

print 'Starting'

vicon = ViconNexus.ViconNexus()

subject = vicon.GetSubjectNames()[0]

markers = vicon.GetMarkerNames(subject)

frames = vicon.GetFrameCount()

# Get data from nexus
print 'Populating data matrix'
rawData = np.zeros((frames,len(markers)*3))
for i in range(0,len(markers)):
	rawData[:,3*i-3], rawData[:,3*i-2], rawData[:,3*i-1], E = vicon.GetTrajectory(subject,markers[i])
	
	rawData[E==0,3*i-3] = 0;
	rawData[E==0,3*i-2] = 0;
	rawData[E==0,3*i-1] = 0;

# Run low dimensional smoothing
Y = smooth.smooth(rawData)

print 'Writing new trajectories'
#Create new smoothed trjectories
for i in range(0,len(markers)):
	E = np.ones((len(E),1)).tolist();
	vicon.SetTrajectory(subject,markers[i],Y[:,3*i-3].tolist(),Y[:,3*i-2].tolist(),Y[:,3*i-1].tolist(),E)
	
print 'Done'	
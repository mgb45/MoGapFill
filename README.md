# MoGapFill
Low dimensional Kalman smoother that fills gaps in motion capture marker trajectories

This repository includes Matlab and Python implementations of a gap filling algorithm (http://dx.doi.org/10.1016/j.jbiomech.2016.04.016) that smooths trajectories in low dimensional subspaces, together with a Python plugin for Vicon Nexus.

# How to use

Matlab - run testSmoother.m, you'll see a visualisation of the code in action on raw data (https://www.c3d.org/sampledata.html)

Python - run testSmoother.py, you'll get an output file out.csv, containing the data with gaps closed

Nexus 2.1.1 - install a suitable Python environment (I used Anaconda Python), make sure .py extensions are set to open with Python. Then, apply the viconMarkerCompletion.py script in Nexus, like this https://youtu.be/_pq57P1lKGg

Note that the method requires that at least one marker be present at all times for inference, so will fail to fill gaps when all markers go missing.

Hope it helps.




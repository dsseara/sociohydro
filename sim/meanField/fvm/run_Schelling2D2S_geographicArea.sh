#!/bin/bash

datafile=/Users/danny/code/uchicago/sociohydro/data/raw/gridded/Illinois_Cook.hdf5
duration=10
nt=100
timestepper=linear
dt=0.01
kW=1
kB=1
kp=-0.5
km=0.3
nu=0.8
capacityType=local
buffer=5
simplify=2
cellsize=4
savefolder=/Users/danny/Library/CloudStorage/GoogleDrive-dsseara@uchicago.edu/My\ Drive/uchicago/sociohydro/2024-05-03_fipyTest
filename=data

python run_Schelling2D2S_geographicArea.py -datafile "$datafile" -duration "$duration" -nt "$nt" -timestepper "$timestepper" -dt "$dt" -kW "$kW" -kB "$kB" -kp "$kp" -km "$km" -nu "$nu" -capacityType "$capacityType" -buffer "$buffer" -simplify "$simplify" -cellsize "$cellsize" -savefolder "$savefolder" -filename "$filename"
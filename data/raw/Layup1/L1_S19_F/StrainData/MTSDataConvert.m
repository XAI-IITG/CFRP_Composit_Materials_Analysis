%% This code reads in an MTS DATA file and transforms it into the appropiate units

% raw files are binary and are transformed in the mts Labview to txt
% _DAT files are the txt files that this code reads in

% We took data continously during fatigue test
% and each time we stopped we took MTS data upto mean value and back with each rosette, these
% are denoted as _STRAIN_(A,M or S)_DAT

clc 
clear all

data = load('L1_S19_F004_DAT');
num_ply=12;


%Channel 1 = load = data(:,1)
%channel 2 = extensometer = data(;,2)
%channel 3 = diplacement = data(:,3)
%channel 4 = strain 1 = data(:,4)
  %during fatigue test --> 0deg gage of actuators
  %on mean value (_STRAIN_) file --> 0deg gage of corresponding rosette
%channel 5 = strain 2 = data(:,5)
  %during fatigue test --> 0deg gage of middle
  %on mean value (_STRAIN_) file --> 45deg gage of corresponding rosette
%channel 6 = strain 3 = data(:,6)
  %during fatigue test --> 0deg gage of sensors
  %on mean value (_STRAIN_) file --> 90deg gage of corresponding rosette
%channel 7 = strain 4 = data(:,7)

%for load and displacement: X=volts*fullscale/10
% for strain gage measurement:
% strain = 4*Eout/F*(GEexc-2*Eout),
% where Eout is actual reading,
%Excitation used
Eexc=5; 

%Gains used per strain channel
G1=100;
G2=100;
G3=200;
G4=200;
F=2.1; %this comes from the gage manufacturers; it varies from 2.1 - 2.09

%Scales used in MTS machine
fullscaleDisp=5;
fullscaleLoad=10;

%load calcs

Load = data(:,1)*fullscaleLoad/10;
Stress = Load/(.0061*num_ply*6);

%Displacement calcs

Disp = data(:,3)*fullscaleDisp/10;

%strain calcs

strain1 = 4*data(:,4)./(F*(G1*Eexc-2*data(:,4)));
strain2 = 4*data(:,5)./(F*(G2*Eexc-2*data(:,5)));
strain3 = 4*data(:,6)./(F*(G3*Eexc-2*data(:,6)));
strain4 = 4*data(:,7)./(F*(G4*Eexc-2*data(:,7)));
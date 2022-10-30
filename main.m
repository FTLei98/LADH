%% data set
clc
clear all;
rand('seed',sum(100*clock))
%% MIRFlickr
load('MIRFLICKR.mat');
X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
R = randperm(size(L,1));
sampleInds = R(2001:end);
queryInds = R(1:2000);
XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
clear X Y L

%% IAPRTC-12
% load('IAPRTC-12.mat');
% X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
% R = randperm(size(L,1));
% sampleInds = R(2001:end);
% queryInds = R(1:2000);
% XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
% XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
% clear X Y L

%% NUSWIDE10
% load('NUSWIDE10.mat');
% X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
% R = randperm(size(L,1));
% sampleInds = R(2001:end);
% queryInds = R(1:2000);
% XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
% XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
% clear X Y L

%% fast label enhancement (FLE)
distribution = FLE(XTrain,YTrain,LTrain);
%% parameter set
param.mu = 1e2;
param.eta  = 1e-2;
param.iter  = 10;
nbitset     = [8,16,32,64,128];
eva_info    = cell(1,length(nbitset));
%% centralization
XTest = bsxfun(@minus, XTest, mean(XTrain, 1)); XTrain = bsxfun(@minus, XTrain, mean(XTrain, 1));
YTest = bsxfun(@minus, YTest, mean(YTrain, 1)); YTrain = bsxfun(@minus, YTrain, mean(YTrain, 1));
%% kernelization
[XKTrain,XKTest] = Kernelize(XTrain, XTest, 500); [YKTrain,YKTest]=Kernelize(YTrain,YTest, 1000);
XKTest = bsxfun(@minus, XKTest, mean(XKTrain, 1)); XKTrain = bsxfun(@minus, XKTrain, mean(XKTrain, 1));
YKTest = bsxfun(@minus, YKTest, mean(YKTrain, 1)); YKTrain = bsxfun(@minus, YKTrain, mean(YKTrain, 1));

%% LADH
for kk= 1:length(nbitset)
    
param.nbits = nbitset(kk);

% Hash codes learning
[B1, B2, B3] = LADH(XKTrain, YKTrain, LTrain,distribution, param, XKTest, YKTest);

% Retrieval tasks
DHamm = hammingDist(B2, B1);
[~, orderH] = sort(DHamm, 2);
Image_to_Text_MAP = mAP(orderH', LTrain, LTest);
 
DHamm = hammingDist(B3, B1);
[~, orderH] = sort(DHamm, 2);
Text_to_Image_MAP = mAP(orderH', LTrain, LTest);

fprintf('LADH %d bits -- Image_to_Text_MAP: %.4f ; Text_to_Image_MAP: %.4f ; \n',nbitset(kk),Image_to_Text_MAP,Text_to_Image_MAP);

end
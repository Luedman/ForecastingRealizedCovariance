clear all; 
rng('default');
load('MGTimeseries.mat')

%%%% generate the training data

% number of layers
T = 3000;
washout_layers = 1000;
% number of neurons
N = 60;
% input mask
C = randn(N,1)*1.e-1;
I0 = 0;
% regularization strength
lambda = 10^(-2);
% order of expansion in the input signal terms
R = 8;


%%%%%%%%%%%%INPUT AND TEACHING SIGNAL TRAINING%%%%%%%

z = MGTimeseries(1,1:T);
y = MGTimeseries(1,2:T+1);

z_test = MGTimeseries(1,T+1:2*T);
y_test = MGTimeseries(1,T+2:2*T+1);


sequenceLength = 3000;
inputSequence = MGTimeseries(1,1:sequenceLength);
outputSequence = MGTimeseries(1,2:sequenceLength + 1); 

%%%% split the data into train and test

train_fraction = 0.75 ; % use 50% in training and 50% in testing
[trainInputSequence, testInputSequence] = ...
    split_train_test(inputSequence,train_fraction);
[trainOutputSequence,testOutputSequence] = ...
    split_train_test(outputSequence,train_fraction);



%%%% generate an esn 
nInputUnits = 1; nInternalUnits = 100; nOutputUnits = 1; 
% 
esn = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
    'spectralRadius',0.8,'inputScaling',[0.1;0.1],'inputShift',[0;0], ...
    'teacherScaling',[0.3],'teacherShift',[-0.2],'feedbackScaling', 0, ...
    'type', 'plain_esn'); 

%%% VARIANTS YOU MAY WISH TO TRY OUT
% (Comment out the above "esn = ...", comment in one of the variants
% below)

% % Use a leaky integrator ESN
% esn = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
%     'spectralRadius',0.5,'inputScaling',[0.1;0.1],'inputShift',[0;0], ...
%     'teacherScaling',[0.3],'teacherShift',[-0.2],'feedbackScaling', 0, ...
%     'type', 'leaky_esn'); 
% 
% % Use a time-warping invariant ESN (makes little sense here, just for
% % demo's sake)
% esn = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
%     'spectralRadius',0.5,'inputScaling',[0.1;0.1],'inputShift',[0;0], ...
%     'teacherScaling',[0.3],'teacherShift',[-0.2],'feedbackScaling', 0, ...
%     'type', 'twi_esn'); 

% % Do online RLS learning instead of batch learning.
% esn = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
%       'spectralRadius',0.4,'inputScaling',[0.1;0.5],'inputShift',[0;1], ...
%       'teacherScaling',[0.3],'teacherShift',[-0.2],'feedbackScaling',0, ...
%       'learningMode', 'online' , 'RLS_lambda',0.9999995 , 'RLS_delta',0.000001, ...
%       'noiseLevel' , 0.00000000) ; 

esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;

%%%% train the ESN
nForgetPoints = 100 ; % discard the first 100 points
[trainedEsn stateMatrix] = ...
    train_esn(trainInputSequence, trainOutputSequence, esn, nForgetPoints) ; 

%%%% save the trained ESN
% save_esn(trainedEsn, 'esn_narma_demo_1'); 

%%%% plot the internal states of 4 units
nPoints = 200 ; 
plot_states(stateMatrix,[1 2 3 4], nPoints, 1, 'traces of first 4 reservoir units') ; 

% compute the output of the trained ESN on the training and testing data,
% discarding the first nForgetPoints of each
nForgetPoints = 100 ; 
predictedTrainOutput = test_esn(trainInputSequence, trainedEsn, nForgetPoints);
predictedTestOutput = test_esn(testInputSequence,  trainedEsn, nForgetPoints) ; 

% create input-output plots
nPlotPoints = 100 ; 
plot_sequence(trainOutputSequence(nForgetPoints+1:end,:), predictedTrainOutput, nPlotPoints,...
    'training: teacher sequence (red) vs predicted sequence (blue)');
plot_sequence(testOutputSequence(nForgetPoints+1:end,:), predictedTestOutput, nPlotPoints, ...
    'testing: teacher sequence (red) vs predicted sequence (blue)') ; 

%%%%compute NRMSE training error
trainError = compute_NRMSE(predictedTrainOutput, trainOutputSequence); 
disp(sprintf('train NRMSE = %s', num2str(trainError)))

%%%%compute NRMSE testing error
testError = compute_NRMSE(predictedTestOutput, testOutputSequence); 
disp(sprintf('test NRMSE = %s', num2str(testError)))


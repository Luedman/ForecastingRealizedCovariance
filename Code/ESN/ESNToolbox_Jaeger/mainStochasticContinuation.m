clear;

load('MGTimeseries');

trainPercentage = 0.5;
sample = MGTimeseries';
sample = (sample - mean(sample))./std(sample);
%sample = tanh(sample - 1);
[train, test] = split_train_test(sample, trainPercentage);

nInputUnits = 1;
nInternalUnits = 400;
nOutputUnits = 1;
inputScaling = randn(1, nInputUnits);
learningMode = 'offline_singleTimeSeries';
reservoirActivationFunction = 'tanh';
methodWeightCompute = 'pseudoinverse';
spectralRadius = 0.5;
%teacherShift = -0.2;
type = 'plain_esn';

esn = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
    'inputScaling', inputScaling, 'learningMode', learningMode, ...
    'reservoirActivationFunction', reservoirActivationFunction, 'methodWeightCompute', methodWeightCompute, ...
    'spectralRadius', spectralRadius, 'type', type);%, 'teacherShift', teacherShift);
esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;

horizon = 1;
trainInput = train(1:end - horizon);
trainOutput = train(horizon + 1:end);
nForgetPoints = 1000;
[trained_esn, stateCollection] = ...
    train_esn(trainInput, trainOutput, esn, nForgetPoints);

testInput = test(horizon:end - horizon);
testOutput = test(horizon + 1:end);

% compute the output of the trained ESN on the training and testing data,
% discarding the first nForgetPoints of each
nForgetPoints = 1000 ; 
predictedTrainOutput = test_esn(trainInput, trained_esn, nForgetPoints);
predictedTestOutput = test_esn(testInput,  trained_esn, nForgetPoints) ; 

% create input-output plots
nPlotPoints = 100 ; 
plot_sequence(trainOutput(nForgetPoints+1:end,:), predictedTrainOutput, nPlotPoints,...
    'training: teacher sequence (red) vs predicted sequence (blue)');
plot_sequence(testOutput(nForgetPoints+1:end,:), predictedTestOutput, nPlotPoints, ...
    'testing: teacher sequence (red) vs predicted sequence (blue)') ; 

%%%%compute NRMSE training error
trainError = compute_NRMSE(predictedTrainOutput, trainOutput); 
disp(sprintf('train NRMSE = %s', num2str(trainError)))

%%%%compute NRMSE testing error
testError = compute_NRMSE(predictedTestOutput, testOutput); 
disp(sprintf('test NRMSE = %s', num2str(testError)))

%%%%
%%%%
horizonContinue = 10;
outputSequence = continue_esn(trainInput, trained_esn, nForgetPoints, horizonContinue);
%trainInput = predictedTrainOutput;

% for i = 1:10
%     predictedTrainOutput = test_esn(trainInput, trained_esn, 0);%, 'startingState', trainInput(:, 1:end-1));
%     trainInput = [trainInput; predictedTrainOutput(end)];
% end
% create input-output plots
% nPlotPoints = 8 ; 
% plot_sequence(MGTimeseries(:,5001:5010), predictedTrainOutput(end - 9:end), nPlotPoints,...
%     'training: teacher sequence (red) vs predicted sequence (blue)');

plot(MGTimeseries(:,1001:5000 + horizonContinue - 1));hold on; plot(outputSequence(1:end))
%%%%compute NRMSE training error
% Error = compute_NRMSE(predictedTrainOutput(end - 19:end), MGTimeseries(:,3001:3010)); 
% disp(sprintf('train NRMSE = %s', num2str(trainError)))

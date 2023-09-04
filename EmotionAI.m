%% See 'audioFeatureExtractor', 'Classify Sound Using Deep Learning', 'Speech Emotion Recognition', 'Speech Command Recognition Using Deep Learning', 'List of Deep Learning Layers', 'reluLayer', 'Classification with Imbalanced Data', 'Record and Play Audio'

%% Finding Data
% 44th and 114th files (and possibly some in between) of any of the folders may be corrupted
% Manually removed 73 files in this range from each folder (Now in Bad Files folder)
auds=audioDatastore('Emotions','IncludeSubfolders',true,'FileExtensions','.wav','LabelSource','foldernames');
[audioValidate,audioTest,audioTrain]=splitEachLabel(auds,0.05,0.1,'randomized');

%% audioFeatureExtractor
overlapLength = 0;
afe = audioFeatureExtractor( ...
    'OverlapLength',overlapLength, ...
    'gtcc',true, ...
    'gtccDelta',true, ...
    'mfccDelta',true, ...
    'SpectralDescriptorInput','melSpectrum', ...
    'spectralCrest',true);
fs=afe.SampleRate;

%% Extracting Features of audioTrain
audioTrainTall=tall(audioTrain);
fTrainTall=cellfun(@(x)extract(afe,x),audioTrainTall,"UniformOutput",false);
fTrainTall=cellfun(@(x)x',fTrainTall,"UniformOutput",false);
fTrain=gather(fTrainTall);
numFiles=numel(fTrain);
[numFeatures1,numHops1,numChannels1]=size(fTrain{1});
[numFeatures2,numHops2,numChannels2]=size(fTrain{2});

%% Extracting Features of audioTest
audioTestTall=tall(audioTest);
fTestTall=cellfun(@(y)extract(afe,y),audioTestTall,"UniformOutput",false);
fTestTall=cellfun(@(y)y',fTestTall,"UniformOutput",false);
fTest=gather(fTestTall);

%% Extracting Features of audioValidate
audioValTall=tall(audioValidate);
fValTall=cellfun(@(z)extract(afe,z),audioValTall,"UniformOutput",false);
fValTall=cellfun(@(z)z',fValTall,"UniformOutput",false);
fVal=gather(fValTall);

%% Layers
layers = [ ...
    sequenceInputLayer(numFeatures1)
    reluLayer
    %dropoutLayer(0.3)
    bilstmLayer(200,"OutputMode","last")
    %dropoutLayer(0.6)
    fullyConnectedLayer(numel(unique(audioTrain.Labels)))
    softmaxLayer
    classificationLayer];

%% Training
options = trainingOptions('adam', ...
    'InitialLearnRate',0.005, ...
    'ValidationData',{fVal,audioValidate.Labels}, ...
    'ValidationFrequency',30, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',64, ...
    'Verbose',false, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress');
[net,info]=trainNetwork(fTrain,audioTrain.Labels,layers,options);
[testPreds,second]=classify(net,fTest);
confusionchart(audioTest.Labels,testPreds);
%save('DLEmotionsTrained.mat');

%% Emotest
adr=audioDeviceReader(fs);
setup(adr)
fileWriter=dsp.AudioFileWriter('mySpeech.wav','FileFormat','WAV');
disp('Start')
time=1;
tic
while toc &lt; 30
    fileWriter=dsp.AudioFileWriter('mySpeech.wav','FileFormat','WAV');
    while toc &lt; time
        acquiredAudio = adr();
        fileWriter(acquiredAudio);
    end
    time=time+2;
    y=audioread('mySpeech.wav');
    z=(extract(afe,y))';
    mood=classify(net,z);
    disp(['You are feeling ',char(mood(1))]);
end
disp('Stop')
release(adr)
release(fileWriter)
%=================SPECCH2FEATS===============
load('workspaceVars.mat')

dataDirTemp=dir(fullfile('fullConversation', '*.flac'));
dataDir=dataDirTemp.folder;
lenDataTrain = 1;%length(y);
ads = audioDatastore(dataDir, 'IncludeSubfolders', true,...
'FileExtensions', '.flac',...
'LabelSource','foldernames')
[trainDatastore, testDatastore] = splitEachLabel(ads,1);% For 80% training 0.8

addpath('VU Task - App\SpeakerIdentificationUsingPitchAndMFCCExample')

features = cell(lenDataTrain,1);
for i = 1:lenDataTrain
    [dataTrain, infoTrain] = read(trainDatastore);
    dataTrain(:,2)=[];
    features{1} = HelperComputePitchAndMFCC(dataTrain,infoTrain);
end
features = vertcat(features{:});
features = rmmissing(features);

% app.FLabel.Text=string('F:eatures:Pitch, MFCC1, ...,MFCC13.');
% drawnow
save('workspaceVars.mat')
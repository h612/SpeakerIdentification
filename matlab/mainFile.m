% For example, typical compact disks use a sample rate of 44,100 hertz and a 
% 16-bit depth. Create an audiorecorder object to record in stereo (two channels)
% with those settings:


% =================	MEETING RECORDING==============
Fs=8300;
myRecObj = audiorecorder(Fs, 16, 2);
timeOfRecording=5;%60*6
disp('Start speaking.')
recordblocking(myRecObj, timeOfRecording);
%disp('');
%app.StatusLabel.Text='Status: End of Recording.';
drawnow
[y] = getaudiodata(myRecObj);
audiowrite('fullConversation\sound_meeting1.wav',y,Fs);%Fs=9100----for test -sound_meetingtest
audiowrite('fullConversation\sound_meeting1.flac',y,Fs);
sound(y)
plot(y);
drawnow

% =================SPEECH2TEXT==============
save('workspaceVars.mat')

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


% %=================FEATS2CLASSII=FIER===============
% 
% 
load('workspaceVars.mat')
inputTable = features;
predictorNames = features.Properties.VariableNames;
predictors = inputTable(:, predictorNames(2:15));
response = inputTable.Label;
% %-------------------with som
% net = selforgmap([10 10]);% #Use Self-Organizing Map to Cluster Data
% net = train(net,predictors{:,:});
% view(net)
% y1 = net(predictors{:,:});
% [iy1 iy2]=find(y1>0);
% %-------------------with kmeans
 [idx, C] = kmeans(predictors{:,:}, 3)
 X=predictors{:,:};
figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids'
hold off
% %-------------------with Gaussian Mixture Model
options = statset('MaxIter',2000); 
gmfit = fitgmdist(X,3,...
            'SharedCovariance',true,'Options',options);
clusterX = cluster(gmfit,X);
k = 3;
figure;
h1 = gscatter(X(:,1),X(:,2),clusterX);
hold on;

plot(gmfit.mu(:,1),gmfit.mu(:,2),'kx','LineWidth',2,'MarkerSize',10)
        legend(h1,{'1','2','3'});
      save('workspaceVars.mat')
      saveas(gcf,'trainedNet.jpg')
      im1=imread('trainedNet.jpg');
%---------------------------

load('workspaceVars.mat')
Fs=8300;
myRecObj = audiorecorder(Fs, 16, 2);
timeOfRecording=5;%60*6
disp('Start speaking.')
recordblocking(myRecObj, timeOfRecording);
%disp('');
% app.StatusLabel_2.Text='Status: End of Recording.';
% drawnow
[y2] = getaudiodata(myRecObj);
audiowrite('sound_meeting_LiveTest.wav',y2,Fs);%Fs=9100----for test -sound_meetingtest
audiowrite('sound_meeting_LiveTest.flac',y2,Fs);
sound(y2)
plot(y2);
drawnow
% app.StatusLabel_2.Text='Status: Plotting Speech chart.';
% drawnow
audioDir='sound_meeting_LiveTest';

%------------- AUDIO DATASTORE WITH LABEL---------------------------
ads = audioDatastore(audioDir, 'IncludeSubfolders', true,...
'FileExtensions', '.flac',...
'LabelSource','foldernames')
%------------- SPLIT DATA TO TEST ONLY---------------------------
[trainDatastore1, testDatastore1] = splitEachLabel(ads,0.01);
%reset(trainDatastore);
%------------- EXTRACT FEATURES ---------------------------
lenDataTrain1 = length(testDatastore1.Files);
addpath('SpeakerIdentificationUsingPitchAndMFCCExample')

features1 = cell(lenDataTrain1,1);
for i = 1:lenDataTrain1
    [dataTrain1, infoTrain1] = read(testDatastore1);
    features1{i} = HelperComputePitchAndMFCC(dataTrain1,infoTrain1);
end
features1 = vertcat(features1{:});
features1 = rmmissing(features1);
%------------- TEST ON PRETRAINED MODEL---------------------------
inputTable1 = features1;
predictorNames1 = features1.Properties.VariableNames;
predictors1 = inputTable1(:, predictorNames1(2:15));
response = inputTable1.Label;
predictors1 = splitvars(predictors1)
predictors1(:,2)=[];
% gmfit = fitgmdist(X,3,...
%             'SharedCovariance',true,'Options',options);
clusterX1 = cluster(gmfit,predictors1{:,:})
most_freq_cluster=mode(clusterX1);
disp(sprintf('Person speaking is :%d',most_freq_cluster));
        
save('workspaceVars.mat')

load('workspaceVars.mat')

transcriber = speechClient('Google','languageCode','en-US','maxAlternatives',3);
[speech,SampleRate] = audioread('VU Task - App\testAudio\Markus 1\speaker_conf3_8.wav');

% % seperatedSoundSignal = reshape(speech,500,((length(speech)/358)));
% % sound(seperatedSoundSignal(:,5))
text = speech2text(transcriber,speech(500:1000,1),SampleRate,'HTTPTimeOut',25);

T = table(most_freq_cluster,text,  'VariableNames', {  'Person ','Text'} )
% Write data to text file
T1 = splitvars(T)
writetable(T1, 'Speech2Text_PersonInfo.txt')

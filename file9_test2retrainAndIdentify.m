% GMM MODEL gmfit clusterX
% kmeans idx
% SOM y1 = net(predictors{:,:}); 

% predictors.Labels=clusterX;
% figure;plot(clusterX);hold on;plot(idx)
% [y,Fs] = audioread('sound_meeting.flac');
% sound(y)

% 
% recObj = audiorecorder
% i=int8(1);
% Fs = 8300;
% while(i<=25)
%     disp(sprintf('Start speaking: %d',i))
%     recordblocking(recObj, 10);
%     disp('End of Recording.');
% 
% 
%    % play(recObj);
% 
%     % Store data in double-precision array, y.
% 
%     y = getaudiodata(recObj);
%     
%     audiowrite(sprintf('speaker%d.wav',i), y, Fs)
%     % Plot the audio samples.
% 
%     plot(y);
%     i=i+1;
% end

%------------- TEST AUDIO CLIPS STORED HERE---------------------------
% dataDir='C:\Users\HUMA\Documents\MATLAB\VU Task\testAudio';%C:\Users\HUMA\AppData\Local\Temp\an4\wav\flacData
% audioDir='C:\Users\HUMA\Documents\MATLAB\VU Task\testAudio\Markus 1\speaker_conf3_8.flac';
%---------------Get NEW DATA OF AUDIO-------------------
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
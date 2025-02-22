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
app.StatusLabel.Text='Status: End of Recording.';
drawnow
[y] = getaudiodata(myRecObj);
audiowrite('fullConversation\sound_meeting1.wav',y,Fs);%Fs=9100----for test -sound_meetingtest
audiowrite('fullConversation\sound_meeting1.flac',y,Fs);
sound(y)
plot(app.UIAxes,y);
drawnow
app.StatusLabel.Text='Status: Plotting Speech chart.';
drawnow
% =================SPEECH2TEXT==============
% transcriber = speechClient('Google');
% [speech,SampleRate] = audioread('sound_meeting.wav');
% 
% text = speech2text(transcriber,speech(:,1),SampleRate,'HTTPTimeOut',25);
save('workspaceVars.mat')

% %=================SPECCH2FEATS===============
% dataDir=('C:\Users\HUMA\Documents\MATLAB\VU Task\fullConversation\');
% lenDataTrain = 1;%length(y);
% ads = audioDatastore(dataDir, 'IncludeSubfolders', true,...
% 'FileExtensions', '.flac',...
% 'LabelSource','foldernames')
% [trainDatastore, testDatastore] = splitEachLabel(ads,1);% For 80% training 0.8
% 
% addpath('C:\Users\HUMA\Documents\MATLAB\Examples\R2019a\audio_wavelet\SpeakerIdentificationUsingPitchAndMFCCExample')
% 
% features = cell(lenDataTrain,1);
% for i = 1:lenDataTrain
%     [dataTrain, infoTrain] = read(trainDatastore);
%     dataTrain(:,2)=[];
%     features{1} = HelperComputePitchAndMFCC(dataTrain,infoTrain);
% end
% features = vertcat(features{:});
% features = rmmissing(features);
% head(features)


%=================FEATS2CLASSII=FIER===============
% inputTable = features;
% predictorNames = features.Properties.VariableNames;
% predictors = inputTable(:, predictorNames(2:15));
% response = inputTable.Label;
% 
% net = selforgmap([10 10]);% #Use Self-Organizing Map to Cluster Data
% net = train(net,predictors{:,:});
% view(net)
% y1 = net(predictors{:,:});
% %-------------------with kmeans
%  [idx, C] = kmeans(predictors{:,:}, 5)
%  X=predictors{:,:};
% figure;
% plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
% hold on
% plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
% plot(C(:,1),C(:,2),'kx',...
%      'MarkerSize',15,'LineWidth',3) 
% legend('Cluster 1','Cluster 2','Centroids',...
%        'Location','NW')
% title 'Cluster Assignments and Centroids'
% hold off

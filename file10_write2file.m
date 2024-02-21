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

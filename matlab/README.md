# SpeechRecognition
![speechtotext](https://github.com/h612/Speech-recognition/assets/23230497/8e244794-ff24-412e-8aea-0f3bbcf27df7)

#Requirements:
1.	Record the meeting discussion voice
2.	Convert it to text
3.	Recognise the speaking person’s voice (identify the speaker)
4.	Assuming speakers talk in turn; if not, the assistant shall prompt for speakers to take turns
5.	When the recognition confidence is low, prompt the speaker to repeat or confirm what was recorded
Data (Record Audio)
Process Flow: 
Convert to text
·	6 minutes Clustering
o	Time sampling
o	Data Sampling Feature Extraction
o	Unsupervised Trained Model
·	start Listening
·	Identify Cluster number (e.g. cluster 3)
·	IDENTIFY OVERLAP
o	Identified Clusters more than 1 – in a given time
o	Identification score low
	Ask to repeat
Algorithm I
1.	Record Audio
2.	Extract Features
3.	Train an unsupervised model
4.	Record Audio
a.	Extract features
b.	Predict speaker
c.	Convert speech to text
i.	For a poor detection, repeat speech 
1. Repeat 4.
ii.	For a good detection, write to file [Person Name, Text] 
1. Repeat 4.


Observations: 
GoogleAPI has a huge resource of recognizable speech. It can be further explored for better applications and problem solutions.
The duration of speech when the Speech-to-Text function is called is not robust. It is fixed to give the GoogleAPI time to match the current phrase. However, in real-time, the duration of phrases vary, and can’t be limited to a short sentence, such as in a conference. Sampling rate of the signals for people who speak different dialects of a language could also vary.
A network design for speech identification must be robust and not limited to a fixed number of people in a meeting, for instance. There is a lot of room for classification techniques for robust applications, such as , for a meeting which grows in participants without prior notice.
Overlapping speech will result in poor detection score and can be corrected. This can begets the machine to be well trained for identification.
Running the Code in Matlab:
Follow the sequence for running different stages of the flow:
1.	file8_speech2textWithGoogle
2.	file8_1_speech2feats.m
3.	file8_som.m
4.	file9_test2retrainAndIdentify.m
5.	file10_write2file.m



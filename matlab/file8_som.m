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
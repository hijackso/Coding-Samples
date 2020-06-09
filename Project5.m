%Part 1 Adding Momemntum to BackPropXOR, see program for fcn
%tested here
clear;
close all;
inputs = [0 0;0 1; 1 0;1 1];
outputs = [0;1;1;0];
WH=rand(4,2); %B/c 4 nodes, 4 rows of weights with 2 inputs from A and B
WO=rand(1,4); %For output layer, each hidden layer node has 1 asst weight

for i=1:10000 %1000 training trials
  [WH,WO] = BackPropSGDMomentum(WH,WO,inputs,outputs);
end

%Test it, test on all 4 choice since that's not so many
for j=1:4
    x = inputs(j,:)';
    vH=WH*x;
    yH=1./(1+exp(-vH));
    vO=WO*yH;
    yO(j) = 1./(1+exp(-vO));
    if yO(j)>0.6;yO(j)=1;else yO(j)=0;end %thresholds outputs
end
yO;

%Part 2, modifying BackPropSGD to use the Cross-entropy fcn
%Same thing, actual fcn is in a different script
clear;
close all;
inputs = [0 0;0 1; 1 0;1 1];
outputs = [0;1;1;0];
WH=rand(4,2); %B/c 4 nodes, 4 rows of weights with 2 inputs from A and B
WO=rand(1,4); %For output layer, each hidden layer node has 1 asst weight

for i=1:10000 %1000 training trials
  [WH,WO] = BackPropSGDCE(WH,WO,inputs,outputs);
end

%Test it, test on all 4 choice since that's not so many
for j=1:4
    x = inputs(j,:)';
    vH=WH*x;
    yH=1./(1+exp(-vH));
    vO=WO*yH;
    yO(j) = 1./(1+exp(-vO));
    if yO(j)>0.6;yO(j)=1;else yO(j)=0;end %thresholds outputs
end
yO;

%Part 3, using the matlab functions to do it
clear;
close all;

inputs = [0 0;0 1;1 0;1 1];
outputs = [0;1;1;0];

net=feedforwardnet();
net.trainFcn='traingdm';

%Gotta have this shit since there's so few training trials
net.trainparam.goal =0;      % This is our idea goal, which means that we'd like the error to be zero (this may not happen, but it's a goal).
net.divideparam.valratio = 0;   %This means don't set aside part of the trianing data for "validation". The data set is already too small (only 4 trials, or rows, of training).
net.divideparam.trainratio =1;  %This means use all the training data (the four rows) for training
net.divideparam.testratio =0;   %This means don't set aside any of the training data for an independent test 
net.trainparam.min_grad = 1e-100;  %This means that stop training if the slope of the error gradient is near zero (you've almost reached the bottom of the error mountain so you're at the flat part of the path (near the minimum of the error function).  
net.trainparam.showwindow=0;  % Don't show the training window when running the program

net = train(net,inputs',outputs');

%test it
net([0;0])
net([0;1])
net([1;0])
net([1;1])

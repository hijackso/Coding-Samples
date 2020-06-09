%Project 9: training a fcn to recongnize hard written numbers using
%MATLABSs built in functions
%Practice with Conc=voutional Neural Nets (CNNs)

clear;
close all;
clc;
[images,correct]= digitTrain4DArrayData; %Imports training data (28x28x1x5000 = 5000 28x28 grayscale images)

trainagain=input('Would you like to train the network [y or n]? ','s');

while trainagain~='n' && trainagain~='y'  % errortrap the input statement
    trainagain=input('Would you like to train the network again [y or n]? ','s');
end

if strcmpi(trainagain,'y')==1  %if the user wants to retrain the network, execute the following lines down to the "else" statement below.
    %A look at the numbers (Saberi Code, gets random images to show
    perm = randperm(size(images,4),20);  % Randomize the order of images in XTrain
    for i = 1:20
    subplot(4,5,i);
    imshow(images(:,:,:,perm(i)));
    end

    %Make the network by adding in each layer
    layers = [imageInputLayer([28 28 1]) %input layer, sized for 1 input image
              convolution2dLayer(9,20,'Padding','same')   %1st of 3 convolutional layers, 20 dif 9x9 filters, adding the padding line means dont pad edges with 0 values, keeps final image the same size
              reluLayer %1st of 3 corresponding relu layers (to conv layer)
              convolution2dLayer(9,20) %This modifies cell outputs
              reluLayer %activation fcn, updates outputs from conv layer
              convolution2dLayer(9,20)
              reluLayer
              averagePooling2dLayer(2,'Stride',2) %2x2 pooling layer (mean), dec net comp load, adding stride means jump 2 to the right for next pool
              fullyConnectedLayer(10) %feed forward net, 1 HL w/ 100 nodes10 output layer nodes (0-9)
              softmaxLayer
              classificationLayer];
    %Set training parameters, 20 epochs, learning rate =0.1
    options=trainingOptions('sgdm','Plots','training-progress','MaxEpochs',20,'InitialLearnRate',0.1,'Shuffle','every-epoch');
    %Train the net!
    trainedNet = trainNetwork(images,correct,layers,options);

    save digitnet trainedNet; %Saves trained network
    % Test Net!
    figure;
    imshow(images(:,:,:,300));
    [guess, prob]=classify(trainedNet,images(:,:,:,300));
else
    load digitnet;  % If you don't want to retrain the network, then load and use the previously trained network.
end  %if strcmpi

%Test using webcam
%Uses webcame to capture image of a digit
%Code copy and pasted from Saberi

camera = webcam; % Connect to the camera

figure;  % open new figure window
while true    %this is a loop that will go on forever unless you break out of it.
    im = snapshot(camera); % Take a picture
    image(im); % Show the picture
    im=rgb2gray(im); %make the image grayscale (that's what the network is expecting)
    im=round(double(im)/255);  % change from 0-255 integer to 0-1 double (increase contrast by using round)
    im = imresize(im,[28 28]); % Resize the picture for the network you trained (it's expecting a 28x28 image)
    label = classify(trainedNet,im); % Classify the picture.  Type help classify in the command window to get more information about this command.
    title(char(label),'fontsize',18); % Show the class label
    drawnow   %force matlab to immediately display the image and label
end
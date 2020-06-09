%Part1 Project 5
% Same as BackPropSGD but *** Uses Momentum ***

%Useful when there is more than 1 hidden layer node. Only good if there's
%only 1 hidden later. Uses back propegation
%This one works only with 1 output later node

function [WH,WO] = DeltaSGD(WH,WO,X,D) %Saying W = makes w the output
%W= old weights, X=inputs, D=correct ans

alpha=0.5; %learning rate
beta=0.5; %modulates momentum
mh1 = 0; %initial momentum hidden layer is 0
mO = 0; %initial m output layer is 0

[R C]=size(X); %Need to know num rows(R) and col (C) for X

for k=1:R %num R = num training trials, since for loop does for every training trial (R training trials)
   x=X(k,:)'; %lower x is single row of X (matrix of all training data) 
              %Does for every colum in row k
              %Transpose function makes good for dot product (the ')
              %x is set of input for 1 trial, if 3 of em, was x1,x2,x3
   d=D(k); %No semicolon b/c is only 1 col of #s (correct answers), so no need to iterate over all of them
           %One value for each row of xs
           %d is the correct answer
   
   vH=WH*x; %Unmodified values sent to each node in HL
   yH=1./(1+exp(-vH));  %Modified (activation fcn) values of each HL node
   
   vO=WO*yH;   %calcs network output node values, same v from before
   yO = 1./(1+exp(-vO)); %Activation function, same as phi(v, calced for output layer
                       %Recall the . make a new vector that hits every value of v
   eO=d-yO; %network error, correct ans-output ans, don't need a period here b/c substraction, only need for /,*,**
          %This error is for output layer   
   deltaO=yO.*(1-yO).*eO; %lower delta is learnign rule of SGD, same eqn from slides replacing phiv with y, this one is for output layer
   
   eH=WO'*deltaO; %Calcs error of hidden layer outputs, recall this eqn was derived from some fancy math we weren't taught
                  %Note! Need to transpose WO!x
   deltaH=yH.*(1-yH).*eH;
   
   %Calc change in weights for each
   changeWH=alpha*deltaH*x'; %Change in HL weights, note x' --> need to make x (2x1 prior to flip) multipliable by deltaH (4x1), this gives us the 4x2 needed for the HL weights
   WH=WH+changeWH+beta*mh1; %Now can add em together no problem, does does 1 for 1 addition. also adding in momentum
   
   changeWO=alpha*deltaO*yH'; %calcs change in weights, output uses node values at input layer, transpose yH so that change WO matches shape of WO
   WO=WO+changeWO+beta*mO; %Same thing, now can add together no problem, also adding in momentum 
   
   mh1 = changeWH+mh1; %update m for next round
   mO = changeWO+mO;
end
end
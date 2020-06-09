%Project 8, practice with CCNs
pete=imread('http://socsci.uci.edu/~saberi/psych149/project8/images/pete.jpg'); %load image of ptae
fSharp=[0 -1 0   %3x3 filter that will shaprpen the image
       -1 5 -1
       0 -1 0];
fEdge=[-1 -1 -1  %edge detecting filter
       -1  8 -1
       -1 -1 -1];
fBlur=[1 2 1     %3x3 blurring filter
       2 4 2
       1 2 1];
sharp=conv2(pete,fSharp); %convolves image pf peter with the sharpening filter
edge=conv2(pete,fEdge); %convolves image pf peter with the edge filter
blur=conv2(pete,fBlur); %convolves image pf peter with the blurring filter

%Let's compare
subplot(4,1,1); %images are kind of small when do this, can switch to new figures for all if want
imshow(pete);
subplot(4,1,2); %subplot sets up a grid and tells computer what part of grid working in
%figure;
imshow(mat2gray(sharp));
subplot(4,1,3); %automatically updates the subplot grid in the figure, unless you tell it you're wokring in a new figure window
%figure;
imshow(mat2gray(edge));
subplot(4,1,4);
%figure;
imshow(mat2gray(blur));

%Some other stuff you can do 
%Example is with blurred image. 

%Improve result and image quality
figure;
subplot(1,2,1);
imshow(pete);
blur=blur./max(max(blur)); %when you convolve two variables, you could end up with really large numbers.  This line scales back y to a reasonable range.   
blur=imadjust(blur);       % Adjust the intensity values of the image to improve contrast when viewing.
subplot(1,2,2);
imshow(mat2gray(blur));

%Blur the shit out of it, convolve blur many times with the resulting blur
%image
s=size(pete); %use to convolve many times and keep the image the same size as the original
for j=1:50   %start the loop, change change #times it does it
   blur=blur(2:s(1)+1,2:s(2)+1);   %everytime you convolve two matrices, the resultant matrix is just a 
                             %little larger than the original.  This line will crop the size of the 
                             %image to match the original 
   pete=blur;                %Makes it so that the convolving matrice is the newly blurred one
   blur=conv2(pete,fBlur);   % convolves the current pete matrice with the blur filter
end
imshow(mat2gray(blur)); %show blurred out image, whackkkk
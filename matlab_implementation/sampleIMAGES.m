function [patches,selection] = sampleIMAGES(patchsize)
% sampleIMAGES
% Returns 10000 patches for training

load IMAGES;    % load images from disk 

%patchsize = 8*8;  % we'll use 8x8 patches 
numpatches = 10000;

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize*patchsize, numpatches);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data 
%  from IMAGES.  
%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1

[rows,cols,numPhotos] = size(IMAGES);
rng('shuffle');
selection = randi(numPhotos)
for i=1:10000
    %origin of the sample point
    rowSeed = randi(rows-patchsize+1);
    colSeed = randi(cols-patchsize+1);

    samplePatch = IMAGES(rowSeed:rowSeed+patchsize-1,colSeed:colSeed+patchsize-1,selection);
    
    %reshape as vector --> column major order
    samplePatch = samplePatch(:);
    patches(:,i) = samplePatch;
end


%{
inputImage = zeros(64,64*64);

[rows,cols,numPhotos] = size(IMAGES);
selection = 1; count = 0;
for i = 1:8:512
    for j = 1:8:512
        samplePatch = IMAGES(i:i+7,j:j+7, selection); 
        samplePatch = samplePatch(:);
        count = count + 1;
        
        inputImage(:,count) = samplePatch;
    end
end
%}
%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
%for debug of c code
%load matlab_patches.mat;
patches = normalizeData(patches);
selection = selection;
%selection = 3;

%inputImage = normalizeData(inputImage);
%patches = inputImage;
end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end

load W1.mat;
load W2.mat;
load matlab_patches.mat;
M = 10000;
rhoHat = 0;
b1 = zeros(25,1);
b2 = zeros(64,1);
beta = 3.0;
lambda = 0.0001;
sparsityParam = 0.01;
W1grad = zeros(size(W1)); %25x64
W2grad = zeros(size(W2)); %64x25
b1grad = zeros(size(b1)); %25x1
b2grad = zeros(size(b2)); %64x1

cost = 0;
for i =1:M
    %forward propagate
    %from a1 = input to a2
    xM = patches(:,i);%current input
    z2 = W1 * xM + b1;
    a2 = 1./ ( 1 + exp(-z2));
   
    %sparsity parameters
    rhoHat = rhoHat + a2;
end

rhoHat = rhoHat ./ M;

   
for i = 1:M
    
    %forward propagate
    %from a1 = input to a2
    xM = patches(:,i);%current input
    z2 = W1 * xM + b1;
    a2 = 1./ ( 1 + exp(-z2));
    
    %from a2 to a3 = output
    z3 = W2 * a2 + b2;
    a3 = 1./ (1 + exp(-z3));
    
    
    %back propagate
    %a3 -> a2
    d3 = -(xM - a3) .* (a3 .* (1 - a3));
    %d2 = (transpose(W2) * d3) .* (a2 .* (1 - a2));
    d2 = ((W2' * d3) + beta .* (-(sparsityParam./rhoHat)...
         + (1-sparsityParam)./(1-rhoHat))).* (a2 .* (1 - a2));

    %compute partial derivatives
    W2grad = W2grad + d3 * a2';
    b2grad = b2grad + d3;
    W1grad = W1grad + d2 * xM';
    b1grad = b1grad + d2; 
  
    %for calculating cost
    %equiv. to (HwbXi-y)^2
    cost = cost + norm(a3 - xM)^2; 
    
end

%W1grad = [(1/m) \Delta W^{(1)} + \lambda W^{(1)}]
W2grad = W2grad ./ M + lambda .* W2;
b2grad = b2grad ./ M;
W1grad = W1grad ./ M + lambda .* W1;
b1grad = b1grad ./ M;


sparsePen = sparsityParam .* log(sparsityParam./rhoHat) + (1-sparsityParam).*log((1-sparsityParam)./(1-rhoHat));

cost = (cost / (2 * M)) + (lambda / 2 ) * (sum(sum(W1.^2)) + (sum(sum(W2.^2)))) + beta * sum(sparsePen); 
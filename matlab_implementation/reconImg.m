function output = reconImg(image, param)
    b1 = param.b1;
    b2 = param.b2;
    W1 = param.W1;
    W2 = param.W2;

    %out1 = 1./(1 + exp(-(W1 * image + repmat(b1,[1,4096]))));
    %out2 = 1./(1 + exp(-(W2 * out1 + repmat(b2,[1,4096]))));
    out1 = W1 * image + repmat(b1,[1,4096]);
    out2 = W2 * out1 + repmat(b2,[1,4096]);
    
    %z2 = W1 * image + repmat(b1,[1,4096]);
    %a2 = 1./(1 + exp(-z2));
    %z3 = W2 * a2 + repmat(b2,[1,4096]);
    %a3 = 1./(1 + exp(-z3));
    
    
    output = zeros(512);
    count = 0;
    for ii = 1:8:512
        for jj = 1:8:512
            count = count + 1;
            output(ii:ii+7, jj:jj+7) = reshape(out2(:,count),[8,8]);
        end
    end
end
function D = FLE(features1,features2, logical_labels)
%% Random Initialization
alpha =95; beta = 50; lamada = 0.1; % only for MIRFlickr
L = logical_labels';
X1 = features1';
X2 = features2';

[num, dim1] = size(features1);% num is the number of instances; dim means the dimension of features  
[~, dim2] = size(features2);
[~,c] = size(logical_labels);  % c is the number of labels

max_iter = 50;                % the max iteration number

% initialization
U1 = rand(dim1,c); U2 = rand(dim2,c); 
D = rand(c,num);
I = ones(1,c);   J = ones(1,num);  K =ones(c,num);

%% Training
for iter = 1 : max_iter
    % U-step
    [R1,~,R2] = svd(X1*D','econ');
    U1 = R1 * R2;
    [R1,~,R2] = svd(X2*D','econ');
    U2 = R1 * R2;
    clear R1 R2;
    
    % D-step
    D = D.*((U1'*X1+U2'*X2+alpha*L+lamada*(I')*J+beta*L.*D)./(U1'*U1*D+U2'*U2*D+alpha*D+lamada*(I')*I*D));
    
end
D = softmax(D)';
end

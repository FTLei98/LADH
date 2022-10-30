function [B1, B2, B3] = LADH(X, Y, L, D, param, XTest, YTest)

% hash codes learning
% min sum_*||WiXi-V||^2 +mu*||B-V||^2 +||rS-V'B||^2 + eta*R(Wi)(l2,1)
% s.t. VV'=nI_r, V1=0, B\in{-1,1}.
%
% hash function learnng
% min ||B-Pi*Xi||^2 

G= NormalizeFea(D,1);
[n, dX] = size(X);
dY = size(Y,2);
X = X'; Y=Y'; L=L';G=G';D=D';
c = size(L,1);

nbits = param.nbits;
mu = param.mu;
eta = param.eta;

% initialization
B = sign(randn(n, nbits))'; B(B==0) = -1;
V = randn(n, nbits)';

W0 = randn(nbits,c);
W1 = randn(nbits,dX);
W2 = randn(nbits,dY);
On = ones(1,n);

for iter = 1:param.iter  

    % B-step
    B = sign(2*nbits*V*G'*G - nbits*V*On'*On + mu*V);

    % W-step
    P0 = diag(sum(W0.^2,2).^-.5);
    W0 = pinv(2*eye(nbits)+eta*P0)*(2*V*D')/(D*D'+eye(c));
    P1 = diag(sum(W1.^2,2).^-.5);
    W1 = pinv(2*eye(nbits)+eta*P1)*(2*V*X')/(X*X'+eye(dX));
    P2 = diag(sum(W2.^2,2).^-.5);
    W2 = pinv(2*eye(nbits)+eta*P2)*(2*V*Y')/(Y*Y'+eye(dY));
    
    % V-step
     W = W0*D + W1*X +W2*Y + 2*nbits*B*G'*G -nbits*B*On'*On + mu*B;
     W = W';
     Temp = W'*W-1/n*(W'*ones(n,1)*(ones(1,n)*W));
     [~,Lmd,QQ] = svd(Temp); clear Temp
     idx = (diag(Lmd)>1e-4);
     M = QQ(:,idx); M_ = orth(QQ(:,~idx));
     Pt = (W-1/n*ones(n,1)*(ones(1,n)*W)) *  (M / (sqrt(Lmd(idx,idx))));
     P_ = orth(randn(n,nbits-length(find(idx==1))));
     V = sqrt(n)*[Pt P_]*[M M_]';
     V = V';  

end

    B1 = B';		
	B1 = B1>0;
    % Hash functions
    Px = B*X'* pinv(X*X'+0.01*eye(dX));
    B2 = XTest*Px'>0;
    Py = B*Y'* pinv(Y*Y'+0.01*eye(dY));
    B3 = YTest*Py'>0;
    
end


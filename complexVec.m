function p = complexVec(N)
%complexVec Returns complex vector
%   Generates an Nx1 vector of complex variables 
%   drawn randomly from the unit circle.
r=rand(N,1)+rand(N,1)*1i;
p=r./abs(r);
end


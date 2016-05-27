%   generate random instance for constrained, convolved, 
%   complex, linear least squares (3CLLS)

clear;
clc;
tic
m = 2000;
n = 100;
y = complexVec(n);
A = convmtx(y,m);
y = complexVec(n+m-1);
A2 = A;
y2 = y;
A = [real(A),-imag(A);imag(A),real(A)];
y = [real(y);imag(y)];
toc

toc
%%
% % %Smooth the signal y, reals and then imag parts
% % %real part
% % D = [];
% % for i=1:length(y)-2
% % D = [D;full(sparse([1 1 1],[i i+1 i+2],[1 -2 1],1,length(y)))];
% % end
% % % Objective
% % lambda = 2;
% % fun = @(x) twonorm(real(y)-x,2)+lambda*twonorm(D*real(y),2);
% % % Objective Gradient (row vector)
% % grad = @(x) transpose(2*(real(x-y))+2*lambda*D'*D*x);
% % % Initial Guess
% % x0 = zeros(length(y),1);
% % % Options
% % opts = optiset('solver','ipopt','maxiter',2000,'display','iter');
% % % Build OPTI Problem
% % Opt = opti('fun',fun,'grad',grad,'x0',x0,'options',opts)
% % % Solve NLP
% % [x,fval,exitflag,info] = solve(Opt)
% % y = complex(x,imag(y));
% % %imag part
% % D = [];
% % for i=1:length(y)-2
% % D = [D;full(sparse([1 1 1],[i i+1 i+2],[1 -2 1],1,length(y)))];
% % end
% % % Objective
% % fun = @(x) twonorm(imag(y)-x,2)+lambda*twonorm(D*imag(y),2);
% % % Objective Gradient (row vector)
% % grad = @(x) transpose(2*(x-imag(y))+2*lambda*D'*D*x);
% % % Initial Guess
% % x0 = zeros(length(y),1);
% % % Options
% % opts = optiset('solver','ipopt','maxiter',2000,'display','iter');
% % % Build OPTI Problem
% % Opt = opti('fun',fun,'grad',grad,'x0',x0,'options',opts)
% % % Solve NLP
% % [x,fval,exitflag,info] = solve(Opt)
% % y = complex(real(y),x);
% % 
% % %transform to reals
% % A2 = A;
% % y2 = y;
% % A = [real(A),-imag(A);imag(A),real(A)];
% % y = [real(y);imag(y)];
% % toc

%explicit - if we use fast method for inverting banded matrix then we are
%golden
% % D = zeros(length(y)-2,length(y));
% % for i=1:length(y)-2
% % D = [D;full(sparse([1 1 1],[i i+1 i+2],[1 -2 1],1,length(y)))];
% % end
% % lambda = 2*n;
% % y = complex(inv(eye(length(y))+lambda*D'*D)*real(y),inv(eye(length(y))+lambda*D'*D)*imag(y));
%%
% Objective
fun = @(x) twonorm(A*x-y,2);

% Objective Gradient (row vector)
grad = @(x) transpose(-2*A'*y + 2*A'*A*x);

%nonlinear constraints
nlcon = @(x) sum([x(1:m).^2,x(m+1:2*m).^2],2);
nlrhs = 0.3^2*ones(m,1);
nle = -1*ones(m,1);
toc
% Bounds
lb = -inf*ones(2*m,1);
ub = inf*ones(2*m,1);

%jacobian
J = @(x) 2*[spdiags(x(1:m),0,m,m),spdiags(x(m+1:2*m),0,m,m)];
%jacobian structure
Jstr = @() [speye(m),speye(m)];
%hessian of lagrangian $\nabla^2L = \sigma \nabla^2 f + \sum_i\lambda_i \nabla^2 c_i$
H = @(x,sigma,lambda) sparse(tril(sigma*sparse(2*A'*A) + 2*[spdiags(lambda,0,m,m),sparse(m,m);sparse(m,m),spdiags(lambda,0,m,m)]));
%hessian structure
Hstr = @() sparse(tril(ones(2*m)));
   
% Initial Guess
x0 = zeros(2*m,1);
toc
% Options
opts = optiset('solver','ipopt','maxiter',2000,'display','iter');
% % opts = optiset('solver','lbfgsb','display','iter');

% Build OPTI Problem
Opt = opti('fun',fun,'grad',grad,'jac',J,'jacstr',Jstr,'hess',H,'hstr',Hstr,'nlmix',...
            nlcon,nlrhs,nle,'bounds',lb,ub,'x0',x0,'options',opts)
% % Opt = opti('fun',fun,'grad',grad,'nlmix',...
% %             nlcon,nlrhs,nle,'bounds',lb,ub,'x0',x0,'options',opts)
% Solve NLP
toc
[x,fval,exitflag,info] = solve(Opt)
toc

%Evaluate
xNL = x;
fNL = twonorm(A*xNL - y,2);
solutionNL = A2*complex(xNL(1:m),xNL(m+1:2*m));
plot(real(solutionNL))
hold on
plot(real(y2))

%%
%Smooth Reals of Results
% % D = [];
% % for i=1:length(solutionNL)-2
% % D = [D;full(sparse([1 1 1],[i i+1 i+2],[1 -2 1],1,length(solutionNL)))];
% % end
% % 
% % % Objective
% % lambda = 2;
% % fun = @(x) twonorm(real(solutionNL)-x,2)+lambda*twonorm(D*real(solutionNL),2);
% % 
% % % Objective Gradient (row vector)
% % grad = @(x) transpose(2*(x-real(solutionNL))+2*lambda*D'*D*x);
% %    
% % % Initial Guess
% % x0 = zeros(length(solutionNL),1);
% % 
% % toc
% % % Options
% % opts = optiset('solver','ipopt','maxiter',2000,'display','iter');
% % % % opts = optiset('solver','lbfgsb','display','iter');
% % 
% % % Build OPTI Problem
% % Opt = opti('fun',fun,'grad',grad,'x0',x0,'options',opts)
% % % % Opt = opti('fun',fun,'grad',grad,'nlmix',...
% % % %             nlcon,nlrhs,nle,'bounds',lb,ub,'x0',x0,'options',opts)
% % % Solve NLP
% % toc
% % [x,fval,exitflag,info] = solve(Opt)
% % toc
% % solutionNLSmooth = x;
% % plot(solutionNLSmooth)

global n dt data_x H lambda lasso_start
%parameters
n = 5000;
T = n^(1/3);
dt = 0.01;%T/n;
%dx=(theta1-theta2*x)dt + (theta2+theta4*x)^theta5 dW, x0=1;
% simulate n trajectories of this process
theta_true = [1, 0.1, 0, 2, 0.5];
x0 = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MC_N = 500;
% theta_qmle = zeros(MC_N, 5);
% theta_adaptive_lasso = zeros(MC_N, 5);
% theta_classic_lasso = zeros(MC_N, 5);
% Hess = zeros(MC_N,5,5);
% classic_Lasso_count = 0;
% adaptive_Lasso_count = 0;
% k = 1;
while (k <= MC_N)
    data_x = generating_data(x0, theta_true);
    % quasi maximum likelihood estimation
    qmle_start = ones(1,5);
    options1 = optimoptions(@fminunc,'Algorithm','quasi-newton','SpecifyObjectiveGradient',true, 'MaxFunEval',10000,'MaxIter',2000, 'TolFun', 1e-8);
    theta_qmle(k,:)= fminunc(@QMLE, qmle_start, options1);
    
     options2 = optimoptions(@fminunc,'Algorithm','trust-region','SpecifyObjectiveGradient',true,  'MaxFunEval',10000,'MaxIter',2000, 'TolFun', 1e-8);
     options3 = optimoptions(@fminunc,'Algorithm','trust-region','SpecifyObjectiveGradient',true, 'HessianFcn','objective', 'MaxFunEval',10000,'MaxIter',2000, 'TolFun', 1e-8);
    %% penalization terms
    lambda0 = 5;
    delta = 2;
    lasso_start = theta_qmle(k,:);     
    %lasso_start = [2.07, 0.56, 0.11, 1.90, 0.52];
    lambda = lambda0* (abs(lasso_start) .^ (-delta));
    for i =1:5
        if lambda(i)>10000
            lambda(i)=10000;
        end
    end
    H = Hessian(lasso_start);
    Hess(k,:,:) = H;

    [~,r] = chol(H); % make sure H is positive definite
    if (r == 0)
        theta_adaptive_lasso(k,:) = fminunc(@Adaptive_Lasso, lasso_start, options3);
        theta_classic_lasso(k,:) = fminunc(@Classic_Lasso, lasso_start, options2);
        if (theta_classic_lasso(k,3) < 0.0001) && (theta_classic_lasso(k,3) > -0.0001)
            classic_Lasso_count =classic_Lasso_count+1;
        end
        if (theta_adaptive_lasso(k,3) < 0.0001) && (theta_adaptive_lasso(k,3) > -0.0001)
            adaptive_Lasso_count =adaptive_Lasso_count+1;
        end
        k= k+1;
     end
end


%%simulate n trajectories of this process with true parameter vector  using the second Milstein scheme
function data = generating_data(x0 , theta)
global n dt

dt_cal = 0.001;
dt_n = fix(dt/dt_cal);

data = zeros(n+1,1);
data(1) = x0;
count =1;

x_old = x0;
for i = 1 : (n*dt_n)
    Z = normrnd(0,1);
    b = theta(1) - theta(2) *x_old;
    b_x = -theta(2);
    b_xx= 0;
    sigma = (theta(3) + theta(4)*x_old)^theta(5);
    sigma_x = theta(4)*theta(5)*(theta(3) + theta(4)*x_old)^(theta(5)-1);
    sigma_xx = theta(4)^2*theta(5)*(theta(5)-1)*(theta(3) + theta(4)*x_old)^(theta(5)-2);
    
    x_new = x_old + dt_cal*(b - 0.5*sigma*sigma_x) + sigma*Z*sqrt(dt_cal) +0.5*sigma*sigma_x*dt_cal*Z^2+...
                dt_cal^(1.5) * (0.5*b*sigma_x +0.5*b_x*sigma +0.25*sigma^2*sigma_xx)*Z + ...
                dt_cal^2 * (0.5*b*b_x +0.25*b_xx*sigma_xx^2);
    if mod(i,dt_n)==0
        count = count+1;
        data(count)= x_new;
    end
    x_old = x_new;
end

return
end


%% the hessian matrix
function He = Hessian(theta)
global n dt data_x

He = zeros(5,5);
for i =1:n
    b = theta(1) - theta(2)*data_x(i);
    sigma = (theta(3) + theta(4)*data_x(i))^ theta(5);
    b_1 = 1;
    b_2 = -data_x(i);
    sigma_3 = theta(5)* (theta(3) + theta(4)*data_x(i))^ (theta(5)-1);
    sigma_4 = sigma_3 * data_x(i);
    sigma_5 = log(abs(theta(3) + theta(4)*data_x(i))) *sigma;
    
    He(1,1) = He(1,1) + (dt/sigma^2)*b_1;
    He(1,2) = He(1,2) + (dt/sigma^2)*b_2;
    a = 2*(data_x(i+1) -data_x(i) -b*dt) / sigma^3;
    He(1,3) = He(1,3) + a*sigma_3;
    He(1,4) = He(1,4) + a*sigma_4;
    He(1,5) = He(1,5) + a*sigma_5;
    He(2,2) = He(2,2) + (-data_x(i)*dt) / sigma^2 * b_2;
    a = -2*data_x(i)*(data_x(i+1) -data_x(i) -b*dt) / sigma^3;
    He(2,3) = He(2,3) + a*sigma_3;
    He(2,4) = He(2,4) + a*sigma_4;
    He(2,5) = He(2,5) + a*sigma_5;
    a= -(data_x(i+1) -data_x(i) -b*dt)^2 / (sigma^3*dt) + 1/ sigma;
    a2= 3*(data_x(i+1) -data_x(i) -b*dt)^2 / (sigma^4*dt) - 1/ sigma^2;
    He(3,3) = He(3,3) + a* theta(5)*(theta(5)-1)*(theta(3)+theta(4)*data_x(i))^(theta(5)-2) + a2* sigma_3^2;
    He(3,4) = He(3,4) + a* data_x(i)*theta(5)*(theta(5)-1)*(theta(3)+theta(4)*data_x(i))^(theta(5)-2) +a2* sigma_4*sigma_3;
    He(4,4) = He(4,4) + a* data_x(i)^2*theta(5)*(theta(5)-1)*(theta(3)+theta(4)*data_x(i))^(theta(5)-2) +a2* sigma_4^2;
    a= -(data_x(i+1) -data_x(i) -b*dt)^2 / (sigma^2*dt) + 1;
    a2= 2*(data_x(i+1) -data_x(i) -b*dt)^2 / (sigma^3*dt);
    He(3,5) = He(3,5) + a/(theta(3)+theta(4)*data_x(i)) + a2*log(theta(3) + theta(4)*data_x(i))*sigma_3;
    He(4,5) = He(4,5) + a*data_x(i)/(theta(3)+theta(4)*data_x(i)) + a2*log(theta(3) + theta(4)*data_x(i))*sigma_4;
    He(5,5) = He(5,5) + a2*log(theta(3) + theta(4)*data_x(i))* sigma_5 ;
end
He = He+triu(He,1).';
return;
end


%% min fun for the QMLE
function [f, g]=QMLE(x)
global n dt data_x

f = 0;
g = zeros(1,5);

for i =1:n
    b = x(1) - x(2)*data_x(i);
    sigma = (x(3) + x(4)*data_x(i))^x(5);
    f = f + 0.5*(log(sigma^2) + (data_x(i+1)-data_x(i)-b*dt)^2 / (sigma^2*dt) );
    g(1) = g(1) + (data_x(i+1)-data_x(i)-b*dt) / (sigma^2) * (-1);
    g(2) = g(2) + (data_x(i+1)-data_x(i)-b*dt) / (sigma^2) * data_x(i);
    g(3) = g(3) + ( 1/sigma - (data_x(i+1)-data_x(i)-b*dt)^2 / (sigma^3*dt) )  * x(5)*(x(3) + x(4)*data_x(i))^(x(5)-1);
    g(4) = g(4) + ( 1/sigma - (data_x(i+1)-data_x(i)-b*dt)^2 / (sigma^3*dt) )  * data_x(i)* x(5)*(x(3) + x(4)*data_x(i))^(x(5)-1);
    g(5) = g(5) + ( 1 - (data_x(i+1)-data_x(i)-b*dt)^2 / (sigma^2*dt) )  * log(abs( x(3) + x(4)*data_x(i) ));
end

return
end

%% min fun for the adaptive Lasso
function [f, g, h]=Adaptive_Lasso(x)
global H lasso_start lambda

f = (x-lasso_start) * H * (x-lasso_start).' + dot(lambda, abs(x));
g = (x-lasso_start) * (H + H.') +  lambda.* sign(x);
h = 2*H;
return
end

function [f, g ] = Classic_Lasso(x)
global n dt data_x lambda

f = 0;
g = zeros(1,5);
for i =1:n
    b = x(1) - x(2)*data_x(i);
    sigma = (x(3) + x(4)*data_x(i))^x(5);
    f = f + 0.5*(log(sigma^2) + (data_x(i+1)-data_x(i)-b*dt)^2 / (sigma^2*dt) );
    g(1) = g(1) + (data_x(i+1)-data_x(i)-b*dt) / (sigma^2) * (-1);
    g(2) = g(2) + (data_x(i+1)-data_x(i)-b*dt) / (sigma^2) * data_x(i);
    g(3) = g(3) + ( 1/sigma - (data_x(i+1)-data_x(i)-b*dt)^2 / (sigma^3*dt) )  * x(5)*(x(3) + x(4)*data_x(i))^(x(5)-1);
    g(4) = g(4) + ( 1/sigma - (data_x(i+1)-data_x(i)-b*dt)^2 / (sigma^3*dt) )  * data_x(i)* x(5)*(x(3) + x(4)*data_x(i))^(x(5)-1);
    g(5) = g(5) + ( 1 - (data_x(i+1)-data_x(i)-b*dt)^2 / (sigma^2*dt) )  * log(abs( x(3) + x(4)*data_x(i) ));
end
f = f + dot(lambda, abs(x));
g = g +  lambda.* sign(x);
end



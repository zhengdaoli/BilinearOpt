function [x,y,resh] = mysaddle(B,c,d,alg,tol,maxiter)
%
% Solve minimax problem:
% min_x max_y f(x,y) = x'By + c'x + d'y
% Inputs:
% (B,c,d) is the problem data where B is m by n (m <= n) and d is in
% the range of B'. Parameter alg specifies one of the 3 algorithms:
% alg = 1: optimistic gradient descent ascent
% alg = 2: extra-gradient method
% alg = 3: proximal point algorithm
% Parameters tol and maxiter are tolerance and maximum iteration number.
% Outputs:
% (x,y) are computed solution and resh stores the iteration history of
% absolute residual norms of the gradient of f(x,y) at each iteration.
    if alg == 1
        [x, y, resh] = ogda(B,c,d,tol, maxiter);
        disp("resh outside: \n");
        disp(size(resh));
    elseif alg == 2
        [x, y, resh] = eg(B,c,d,tol, maxiter);
    else
        [x, y, resh] = ppa(B,c,d,tol, maxiter);
    end
end
%% ogda
function [x, y, resh] = ogda(B,c,d,tol, maxiter)
    [m, n] = size(B);
    resh = cell(maxiter, 1);
    res_pre = 0;
    alpha = 1/(40 * sqrt(eigs(B'*B, 1)));
%     disp("alphaaaa");
%     disp(alpha);
    alpha = 0.0025; % type=1
%     alpha = 0.085;    % type=3
%     alpha = 1.1;    % type=2
%     alpha = 10; % test1
    disp("alpha:");
    disp(alpha);
    beta = 0.6 * alpha;
    beta1 = 0.6 * alpha;
    x = zeros(m, 1); y = zeros(n, 1);
    x_k = x; y_k = y;
    x_km1 = x; y_km1 = y;
    iter = 1;
    while 1
        %% get gradient w.r.t x, y at k, k-1 step:
        gxk = gradx(B, y_k, c);
        gyk = grady(B, x_k, d);
        gxkm1 = gradx(B, y_km1, c);
        gykm1 = grady(B, x_km1, d);
        %% get x_k+1, y_k+1:
        x_kp1 = x_k - alpha*gxk + beta *gxkm1;
        y_kp1 = y_k + alpha*gyk - beta1 *gykm1;
        
        res = norm(gxk) + norm(gyk);
        resh{iter, 1} = res;
        %% print info
        printinfo(res, res_pre, iter);
        res_pre = res;
        %% stoping criteria:
        if res_pre <= tol || iter >= maxiter
            x = x_kp1;
            y = y_kp1;
            resh = cell2mat(resh);
            break
        end
                
        %% reset x, y 
        x_km1 = x_k;
        y_km1 = y_k;
        x_k = x_kp1;
        y_k = y_kp1;
        iter = iter + 1;
    end
end

%% eg
function [x, y, resh] = eg(B,c,d,tol, maxiter)
    [m, n] = size(B);
    resh = cell(maxiter, 1);
    res_pre = 0;
    x = zeros(m,1); y = zeros(n, 1);
    x_k = x; y_k = y;
    eta = 1/(2*sqrt(eigs(B'*B, 1)));
    eta = 11; % test1
%     eta = 0.07; % test2 type=3;
%     eta = 0.7; % test2 type=2;
    eta = 0.002; % test2 type=1;
    fprintf("eta of eg alg: %d \n", eta);
    iter = 1;
    while 1
        %% get gradient w.r.t x, y at k, k-1 step:
        gxk = gradx(B, y_k, c);
        gyk = grady(B, x_k, d);
        
        x_kh = x_k - eta * gxk;
        y_kh = y_k + eta * gyk;
        
        gxkh = gradx(B, y_kh, c);
        gykh = grady(B, x_kh, d);
        
        x_kp1 = x_k - eta * gxkh;
        y_kp1 = y_k + eta * gykh;
        
        res = norm(gxk) + norm(gyk);
        resh{iter, 1} = res;
        %% print
        printinfo(res, res_pre, iter);
        res_pre = res;
        %% stoping criteria:
        if res_pre <= tol || iter > maxiter
            x = x_kp1;
            y = y_kp1;
            resh = cell2mat(resh);
            break
        end
        
        %% reset x, y
        x_k = x_kp1;
        y_k = y_kp1;
        iter = iter + 1;
    end
end
%% pp
function [x, y, resh, maxiter] = ppa(B,c,d,tol, maxiter)
    [m, n] = size(B);
    resh = cell(maxiter, 1);
    res_pre = 0;
    lambda_x = 100;
    lambda_y =  100;
    fprintf("PPA lambdax=lambday=eta=100 \n");
%     x = 0.01 * ones(m, 1); y = 0.01 * ones(n, 1);
    x= zeros(m, 1); y = zeros(n, 1);
    x_k = x; y_k = y;
    iter = 1;
    lambda2_x = lambda_x^2;
    lambda2_y = lambda_y^2;
    A_x = sparse(speye(m) + lambda2_x * (B * B'));
    p = symamd(A_x);
    using_chol = 0;
    R = chol(A_x(p, p));
    while 1
        b_x = x_k - lambda_x * B * y_k - lambda2_x * B * d - lambda_x*c;
        if using_chol
            %% solve by chol
            x_kp1_tmp(p) = R\(R'\b_x(p));
            x_kp1 = x_kp1_tmp';
        else
            %% solve by pcg
            [x_kp1,~,~,~] = pcg(A_x, b_x, 1e-11, 500);
        end
        y_kp1 = y_k + lambda_y*(B' * x_kp1 + d);
        gxk = gradx(B, y_kp1, c);
        gyk = grady(B, x_kp1, d);
        res = norm(gxk) + norm(gyk);
        resh{iter, 1} = res;
        %% print
        printinfo(res, res_pre, iter);
        res_pre = res;
        %% stoping criteria:
        if res_pre <= tol || iter > maxiter
            x = x_kp1;
            y = y_kp1;
            resh = cell2mat(resh);
            break
        end
                
        %% reset x, y 
        x_k = x_kp1;
        y_k = y_kp1;
        iter = iter + 1;
    end

end

function gx=gradx(B,y,c)
    gx = B * y + c;
end

function gy=grady(B,x,d)
    gy = B' * x + d;
end

function [] = printinfo(res, res_pre, iter)
    if iter > 1
        if iter <= 100
            printt(res, res_pre, iter, 10)
        elseif iter <= 1000
            printt(res, res_pre, iter, 100)
        else
            printt(res, res_pre, iter, 1000)
        end
    end
end

function [] = printt(res, res_pre, iter, num)
    if mod(iter, num) == 0
        ratio = res/res_pre;
        fprintf("my: iter    %d: res = %f  ratio = %f \n", iter, res, ratio);
    end
end




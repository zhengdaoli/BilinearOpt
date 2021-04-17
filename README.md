# BilinearOpt

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

function [x, flag, iter, resids] = gmresMILU_MGS(A, b, ...
    M, restart, rtol, maxit, x0, verbose, nthreads)
%gmresMILU_MGS Kernel of gmresMILU using modified Gram-Schmidt
%
%   x = gmresMILU_MGS(A, b, M, restart, rtol, maxit, x0, verbose, nthreads)
%     when uncompiled, call this kernel function by passing the M
%     struct returned by MILUfactor
%
%   [x, flag, iter, resids] = gmresMILU_MGS(...)
%
% See also: gmresMILU, gmresMILU_CGS, gmresMILU_HO

% Note: The algorithm uses the modified Gram-Schmidt orthogonalization.
% It has less parallelism than classical Gram-Schmidt but is more stable.
% It is also less stable than the Householder algorithm.

%#codegen -args {crs_matrix, m2c_vec, MILU_Prec, int32(0), 0., int32(0),
%#codegen m2c_vec, int32(0), int32(0)}

n = int32(size(b, 1));

% If RHS is zero, terminate
beta0 = sqrt(vec_sqnorm2(b));
if beta0 == 0
    x = zeros(n, 1);
    flag = int32(0);
    iter = int32(0);
    resids = 0;
    return;
end

% Number of inner iterations
if restart > n
    restart = n;
elseif restart <= 0
    restart = int32(1);
end

% Determine the maximum number of outer iterations
max_outer_iters = int32(ceil(double(maxit)/double(restart)));

% Initialize x
if isempty(x0)
    x = zeros(n, 1);
else
    x = x0;
end

% Local linear system
y = zeros(restart+1, 1);
R = zeros(restart, restart);

% Orthognalized Krylov subspace
Q = zeros(n, restart);

% Preconditioned subspace
Z = zeros(n, restart);

% Given's rotation vectors
J = zeros(2, restart);

% Buffer spaces
v = zeros(n, 1);
if ~isempty(coder.target)
    y2 = zeros(M(1).negE.nrows, 1);
end

if nargout > 3
    resids = zeros(maxit, 1);
end

flag = int32(0);
iter = int32(0);
resid = 1;
for it_outer = 1:max_outer_iters
    % Compute the initial residual
    if it_outer > 1 || vec_sqnorm2(x) > 0
        v = crs_prodAx(A, x, v, nthreads);
        v = b - v;
    else
        v = b;
    end

    beta2 = vec_sqnorm2(v);
    beta = sqrt(beta2);

    % The first Q vector
    y(1) = beta;
    Q(:, 1) = v / beta;

    j = int32(1);
    while true
        w = Q(:, j);
        % Compute the preconditioned vector and store into v
        if isempty(coder.target)
            w = ILUsol(M, w);
        else
            [w, v, y2] = MILUsolve(M, w, v, y2);
        end

        % Store the preconditioned vector
        Z(:, j) = w;
        v = crs_prodAx(A, w, v, nthreads);

        % Perform Gram-Schmidt orthogonalization and store column of R in w
        for k = 1:j
            w(k) = v' * Q(:, k);
            v = v - w(k) * Q(:, k);
        end

        vnorm2 = vec_sqnorm2(v);
        vnorm = sqrt(vnorm2);
        if j < restart
            Q(:, j+1) = v / vnorm;
        end

        %  Apply Given's rotations to w.
        for colJ = 1:j-1
            tmpv = w(colJ);
            w(colJ) = conj(J(1, colJ)) * w(colJ) + conj(J(2, colJ)) * w(colJ+1);
            w(colJ+1) = - J(2, colJ) * tmpv + J(1, colJ) * w(colJ+1);
        end

        %  Compute Given's rotation Jm.
        rho = sqrt(w(j)'*w(j)+vnorm2);
        J(1, j) = w(j) ./ rho;
        J(2, j) = vnorm ./ rho;
        y(j+1) = - J(2, j) .* y(j);
        y(j) = conj(J(1, j)) .* y(j);
        w(j) = rho;
        R(1:j, j) = w(1:j);

        resid_prev = resid;
        resid = abs(y(j+1)) / beta0;
        if resid >= resid_prev * (1 - 1.e-8)
            flag = int32(3); % stagnated
            break
        elseif iter >= maxit
            flag = int32(1); % reached maxit
            break
        end
        iter = iter + 1;

        if verbose > 1
            m2c_printf('At iteration %d, relative residual is %g.\n', iter, resid);
        end

        % save the residual
        if nargout > 3
            resids(iter) = resid;
        end

        if resid < rtol || j >= restart
            break;
        end
        j = j + 1;
    end

    if verbose == 1 || verbose >1 && flag
        m2c_printf('At iteration %d, relative residual is %g.\n', iter, resid);
    end

    % Compute correction vector
    y = backsolve(R, y, j);
    for i = 1:j
        x = x + y(i) * Z(:, i);
    end

    if resid < rtol || flag
        break;
    end
end

if nargout > 3
    resids = resids(1:iter);
end

if resid <= rtol * (1 + 1.e-8)
    flag = int32(0);
end

end

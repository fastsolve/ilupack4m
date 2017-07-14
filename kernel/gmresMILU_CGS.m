function [x, flag, iter, resids] = gmresMILU_CGS(A, b, ...
    prec, restart, rtol, maxit, x0, verbose, nthreads, param, rowscal, colscal)
%gmresMILU_CGS Kernel of gmresMILU using classical Gram-Schmidt
%
%   x = gmresMILU_CGS(A, b, prec, restart, rtol, maxit, x0, verbose, nthreads)
%     when uncompiled, call this kernel function by passing the prec
%     struct returned by MILUinit
%
%   x = gmresMILU_CGS(A, b, prec, restart, rtol, maxit, x0, verbose, nthreads,
%      param, rowscal, colscal) take the opaque pointers for prec and param
%      and in addition rowscal and colscal in the PREC struct.
%
%   [x, flag, iter, resids] = gmresMILU_CGS(...)
%
% See also: gmresMILU, gmresMILU_MGS, gmresMILU_HO

% Note: The algorithm uses the classical Gram-Schmidt orthogonalization.
% It has more parallelism than modified  Gram-Schmidt but is less stable.
% It is also less stable than the Householder algorithm.

%#codegen -args {crs_matrix, m2c_vec, MILU_Dmat, int32(0), 0., int32(0),
%#codegen m2c_vec, int32(0), int32(0), MILU_Dparam, m2c_vec, m2c_vec}

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
w = zeros(n, 1);
dbuff = zeros(3*n, 1);

if nargout > 3
    resids = zeros(maxit, 1);
end

if ~isempty(coder.target)
    t_prec = MILU_Dmat(prec);
    t_param = MILU_Dparam(param);
    
    need_rowscaling = any(rowscal ~= 1);
    need_colscaling = any(colscal ~= 1);    
end

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
        % Compute the preconditioned vector and store into v
        if isempty(coder.target)
            w = ILUsol(prec, Q(:, j));            
        else
            % We need to perform row-scaling and column scaling.
            % See DGNLilupacksol.c
            if need_rowscaling 
                v = Q(:, j) .* rowscal;
            else
                v = Q(:, j);
            end
            coder.ceval('DGNLAMGsol_internal', t_prec, t_param, ...
                coder.rref(v), coder.ref(w), coder.ref(dbuff));

            if need_colscaling 
                w = w .* colscal;
            end
        end
        
        % Store the preconditioned vector
        Z(:, j) = w;
        v = crs_prodAx(A, w, v, nthreads);

        % Perform classical Gram-Schmidt orthogonalization
        w = v;
        for k = 1:j
            R(k, j) = w' * Q(:, k);
            v = v - R(k, j) * Q(:, k);
        end
        
        vnorm2 = vec_sqnorm2(v);
        vnorm = sqrt(vnorm2);
        if j < restart
            Q(:, j+1) = v / vnorm;
        end

        %  Apply Given's rotations to R(:,j)
        for colJ = 1:j-1
            tmpv = R(colJ, j);
            R(colJ, j) = conj(J(1, colJ)) * R(colJ, j) + conj(J(2, colJ)) * R(colJ+1, j);
            R(colJ+1, j) = - J(2, colJ) * tmpv + J(1, colJ) * R(colJ+1, j);
        end

        %  Compute Given's rotation Jm.
        rho = sqrt(R(j, j)'*R(j, j)+vnorm2);
        J(1, j) = R(j, j) ./ rho;
        J(2, j) = vnorm ./ rho;
        y(j+1) = - J(2, j) .* y(j);
        y(j) = conj(J(1, j)) .* y(j);
        R(j, j) = rho;

        resid = abs(y(j+1)) / beta0;
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

    if verbose == 1
        m2c_printf('At iteration %d, relative residual is %g.\n', iter, resid);
    end

    % Compute correction vector
    y = backsolve(R, y, j);
    for i = 1:j
        x = x + y(i) * Z(:, i);
    end

    if resid < rtol
        break;
    end
end

if nargout > 3
    resids = resids(1:iter);
end

if resid > rtol
    % Did not converge after maximum number of iterations
    flag = int32(1);
else
    flag = int32(0);
end

end

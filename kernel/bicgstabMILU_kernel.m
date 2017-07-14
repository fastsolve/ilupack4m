function [x, flag, iter, resids] = bicgstabMILU_kernel(A, b, ...
    prec, rtol, maxit, x0, verbose, nthreads, param, rowscal, colscal)
%bicgstabMILU_kernel Kernel of bicgstabMILU
%
%   x = bicgstabMILU_kernel(A, b, prec, rtol, maxit, x0, verbose, nthreads)
%     when uncompiled, call this kernel function by passing the prec
%     struct returned by MILUinit
%
%   x = bicgstabMILU_kernel(A, b, prec, rtol, maxit, x0, verbose, nthreads,
%      param, rowscal, colscal) take the opaque pointers for prec and param
%      and in addition rowscal and colscal in the PREC struct.
%
%   [x, flag, iter, resids] = bicgstabMILU_kernel(...)
%
% See also: bicgstabMILU

%#codegen -args {crs_matrix, m2c_vec, MILU_Dmat, 0., int32(0),
%#codegen m2c_vec, int32(0), int32(0), MILU_Dparam, m2c_vec, m2c_vec}

n = int32(size(b, 1));
flag = int32(0);
iter = int32(0);

% If RHS is zero, terminate
bnrm2 = sqrt(vec_sqnorm2(b));
if bnrm2 == 0
    x = zeros(n, 1);
    resids = 0;
    return;
end

% Initialize x
if isempty(x0)
    x = zeros(n, 1);
else
    x = x0;
end

% Buffer spaces
r = zeros(n, 1);
v = zeros(n, 1);
p = zeros(n, 1);
p_hat = zeros(n, 1);
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

% Compute the initial residual
if vec_sqnorm2(x) > 0
    r = crs_prodAx(A, x, r, nthreads);
    r = b - r;
else
    r = b;
end

resid = sqrt(vec_sqnorm2(r)) / bnrm2;
if resid < rtol
    resids = 0;
    return 
end

omega = 1.0;
alpha = 0.0;
rho_1 = 0.0;
r_tld = r;

iter = int32(1);
while true
    rho = (r_tld' * r); % direction vector
    if rho == 0.0
        break
    end
    
    if iter > 1
        beta = (rho / rho_1) * (alpha / omega);
        p = r + beta * (p - omega * v);
    else
        p = r;
    end
    
    % Compute the preconditioned vector and store into v
    if isemedpty(coder.target)
        p_hat = ILUsol(prec, p);
    else
        % We need to perform row-scaling and column scaling.
        % See DGNLilupacksol.c
        if need_rowscaling 
            v = p .* rowscal;
        end
        coder.ceval('DGNLAMGsol_internal', t_prec, t_param, ...
            coder.rref(v), coder.ref(p_hat), coder.ref(dbuff));

        if need_colscaling 
            p_hat = p_hat .* colscal;
        end
    end

    v = crs_prodAx(A, p_hat, v, nthreads);
    alpha = rho / (r_tld' * v);
    x = x + alpha * p_hat;
    s = r - alpha * v;
    snrm = sqrt(vec_sqnorm2(s));

    if snrm < rtol % early convergence check
        resid = snrm / bnrm2;
        resids(iter) = resid;
        break;
    end
    
    % Compute the preconditioned vector and store into v
    if isempty(coder.target)
        p_hat = ILUsol(prec, s);            
    else
        % We need to perform row-scaling and column scaling.
        % See DGNLilupacksol.c
        if need_rowscaling 
            v = s .* rowscal;
        end
        coder.ceval('DGNLAMGsol_internal', t_prec, t_param, ...
            coder.rref(v), coder.ref(p_hat), coder.ref(dbuff));

        if need_colscaling 
            p_hat = p_hat .* colscal;
        end
    end
    
    v = crs_prodAx(A, p_hat, v, nthreads);
    omega = (v' * s) / vec_sqnorm2(v);    
    x = x + omega * p_hat; % update approximation
    
    r = s - omega * v;
    resid = sqrt(vec_sqnorm2(r)) / bnrm2; % check convergence
    resids(iter) = resid;

    if verbose > 1 || verbose > 0 && mod(iter, 30) == 0
        m2c_printf('At iteration %d, relative residual is %g.\n', iter, resid);
    end

    if resid <= rtol
        break
    end
    
    if omega == 0.0
        break
    end
    rho_1 = rho;

    if iter >= maxit
        break
    end
    iter = iter + 1;
end

if nargout > 3
    resids = resids(1:iter);
end

if resid <= rtol % converged
    flag = int32(0);
elseif omega == 0.0 % breakdown
    flag = int32(-2);
elseif rho == 0.0
    flag = int32(-1);
else % no convergence
    flag = int32(1);
end

end

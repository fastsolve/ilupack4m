function [x, flag, iter, resids] = fgmresMILU_kernel(A, b, ...
    prec, restart, rtol, maxit, x0, verbose, param, rowscal, colscal)
%fgmresMILU_kernel The kernel function of fgmresMILU
%
%   x = fgmresMILU_kernel(A, b, prec, restart, rtol, maxit, x0, verbose, param)
%   x = fgmresMILU_kernel(A, b, prec, restart, rtol, maxit, x0, verbose, param, rowscal, colscal)
%
%   [x, flag, iter, resids] = fgmresMILU_kernel(...)
%
% See also: fgmresMILU

%#codegen -args {crs_matrix, m2c_vec, MILU_Dmat, int32(0), 0.,
%#codegen int32(0), m2c_vec, int32(0), MILU_Dparam, m2c_vec, m2c_vec}

n = int32(size(b, 1));

% If RHS is zero, terminate
beta0 = sqrt(vec_sqnorm2(b));
if (beta0 == 0)
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

% Householder matrix
y = zeros(restart+1, 1);
R = zeros(restart, restart);
J = zeros(2, restart);
dx = zeros(n, 1);

% Krylov subspace
V = zeros(n, restart);
% Preconditioned subspace
Z = zeros(n, restart);

% Allocate 3n buffer space for ILUPACK
dbuff = zeros(3*n, 1);

if nargout > 3
    resids = zeros(maxit, 1);
end

iter = int32(0);
resid = 0;

if ~isempty(coder.target)
    t_prec = MILU_Dmat(prec);
    t_param = MILU_Dparam(param);
end

for it_outer = 1:max_outer_iters
    % Compute the initial residual
    if (it_outer > 1) || vec_sqnorm2(x) > 0
        Ax = crs_prodAx(A, x);
        u = b - Ax;
    else
        u = b;
    end
    
    beta2 = vec_sqnorm2(u);
    beta = sqrt(beta2);
    
    % Prepare the first Householder vector
    if (u(1) < 0)
        beta = -beta;
    end
    updated_norm = sqrt(2*beta2+2*real(u(1))*beta);
    u(1) = u(1) + beta;
    u = u / updated_norm;
    
    % The first Householder entry
    y(1) = - beta;
    V(:, 1) = u;
    
    j = int32(1);
    while true
        % Construct the last vector from the Householder reflectors
        
        %  v = Pj*ej = ej - 2*u*u'*ej
        v = -2 * conj(V(j, j)) * V(:, j);
        v(j) = v(j) + 1;
        %  v = P1*P2*...Pjm1*(Pj*ej)
        if isempty(coder.target)
            % This is faster when interpreted
            for i = (j - 1): - 1:1
                v = v - 2 * (V(:, i)' * v) * V(:, i);
            end
        else
            % This is faster when compiled
            for i = (j - 1): - 1:1
                s = conj(V(i, i)) * v(i);
                for k = i + 1:n
                    s = s + conj(V(k, i)) * v(k);
                end
                s = 2 * s;
                
                for k = i:n
                    v(k) = v(k) - s * V(k, i);
                end
            end
        end
        %  Explicitly normalize v to reduce the effects of round-off.
        v = v / sqrt(vec_sqnorm2(v));
        
        % Store the preconditioned vector
        if isempty(coder.target)
            Z(:, j) = ILUsol(prec, v);
        else
            % We need to perform row-scaling and column scaling.
            % Refer to DGNLilupacksol.c
            vscaled = v .* rowscal;
            coder.ceval('DGNLAMGsol_internal', t_prec, t_param, ...
                coder.rref(vscaled), coder.ref(dx), coder.ref(dbuff));
            Z(:, j) = dx .* colscal;
        end
        w = crs_prodAx(A, Z(:, j));
        
        % Orthogonalize the Krylov vector
        %  Form Pj*Pj-1*...P1*Av.
        if isempty(coder.target)
            % This is faster when interpreted
            for i = 1:j
                w = w - 2 * (V(:, i)' * w) * V(:, i);
            end
        else
            % This is faster when compiled
            for i = 1:j
                s = conj(V(i, i)) * w(i);
                for k = i + 1:n
                    s = s + conj(V(k, i)) * w(k);
                end
                s = s * 2;
                
                for k = i:n
                    w(k) = w(k) - s * V(k, i);
                end
            end
        end
        
        % Update the rotators
        %  Determine Pj+1.
        if j < n
            %  Construct u for Householder reflector Pj+1.
            u(j) = 0;
            u(j+1) = w(j+1);
            alpha2 = conj(w(j+1)) * w(j+1);
            for k = j + 2:n
                u(k) = w(k);
                alpha2 = alpha2 + conj(w(k)) * w(k);
            end
            
            if alpha2 > 0
                alpha = sqrt(alpha2);
                if u(j+1) < 0
                    alpha = -alpha;
                end
                if j < restart
                    updated_norm = sqrt(2*alpha2+2*real(u(j+1))*alpha);
                    u(j+1) = u(j+1) + alpha;
                    for k = j + 1:n
                        V(k, j+1) = u(k) / updated_norm;
                    end
                end
                
                %  Apply Pj+1 to v.
                w(j+2:end) = 0;
                w(j+1) = - alpha;
            end
        end
        
        %  Apply Given's rotations to the newly formed v.
        for colJ = 1:j - 1
            tmpv = w(colJ);
            w(colJ) = conj(J(1, colJ)) * w(colJ) + conj(J(2, colJ)) * w(colJ+1);
            w(colJ+1) = - J(2, colJ) * tmpv + J(1, colJ) * w(colJ+1);
        end
        
        %  Compute Given's rotation Jm.
        if (j ~= length(w))
            rho = sqrt(w(j)'*w(j)+w(j+1)'*w(j+1));
            J(1, j) = w(j) ./ rho;
            J(2, j) = w(j+1) ./ rho;
            y(j+1) = - J(2, j) .* y(j);
            y(j) = conj(J(1, j)) .* y(j);
            w(j) = rho;
            w(j+1) = 0;
        end
        
        R(:, j) = w(1:restart);
        
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
    
    if verbose > 0
        m2c_printf('At iteration %d, relative residual is %g.\n', iter, resid);
    end
    
    % Correction
    y = backsolve(R, y, j);
    
    dx = y(j) * Z(:, j);
    for i = j - 1:-1:1
        dx = dx + y(i) * Z(:, i);
    end
    x = x + dx;
    
    if (resid < rtol)
        break;
    end
end

if nargout > 3
    resids = resids(1:iter);
end

if resid > rtol
    % Did not converge after maximum number of iterations
    flag = int32(3);
else
    flag = int32(0);
end

end

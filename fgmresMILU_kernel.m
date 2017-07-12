function [x, iter, resids] = fgmresMILU_kernel(A, b, prec, x0, options)
%fgmresMILU_kernel Kernel function of fgmresMILU_kernel
%   [x, iter, resids] = fgmresMILU_kernel(A, b, prec, x0, options)
%
%
% See also: fgmresMILU

n = int32(size(b, 1));

% Norm of the RHS
beta0 = norm(b);
if (beta0 == 0)
    x = zeros(n, 1);
    iter = int32(0);
    resids = 0;
    return;
end

% Number of inner iterations
restart = int32(options.nrestart);
if (restart > n)
    restart = n;
end

% Maximum number of outer iterations
max_outer_iters = int32(ceil(double(options.maxit)/double(restart)));

% Initialize x
if (isempty(x0))
    x = zeros(n, 1);
else
    x = x0;
end

% Stopping tolerance
tol_exit = options.restol;

% Householder data
W = zeros(restart+1, 1);
R = zeros(restart, restart);
J = zeros(2, restart);

% Krylov subspace
V = zeros(n, restart);
% Preconditioned subspace
Z = cell(restart, 1);

if (nargout > 2)
    resids = zeros(options.maxit, 1);
end

iter = 0;
for it_outer = 1:max_outer_iters
    % Compute the initial residual
    if (it_outer > 1) || (norm(x) > 0)
        if isstruct(A)
            Ax = crs_prodAx(A, x);
        else
            Ax = A * x;
        end
        u = b - Ax;
    else
        u = b;
    end
    beta = norm(u);
    
    % Prepare the first Householder vector
    if (u(1) < 0)
        beta = -beta;
    end
    u(1) = u(1) + beta;
    u = u / norm(u);
    % The first Householder entry
    W(1) = - beta;
    V(:,1) = u;
    
    for j = 1:restart
        % Construct the last vector from the HH storage
        %  Form P1*P2*P3...Pj*ej.
        %  v = Pj*ej = ej - 2*u*u'*ej
        v = -2 * conj(u(j)) * u;
        v(j) = v(j) + 1;
        %  v = P1*P2*...Pjm1*(Pj*ej)
        for i = (j - 1): - 1:1
            v = v - 2 * V(:,i) * (V(:,i)' * v);
        end
        %  Explicitly normalize v to reduce the effects of round-off.
        v = v / norm(v);
        
        % Keep the PrecVec separately
        Z{j} = ILUsol(prec, v);
        if isstruct(A)
            w = crs_prodAx(A, Z{j});
        else
            w = A * Z{j};
        end
        
        % Orthogonalize the Krylov vector
        %  Form Pj*Pj-1*...P1*Av.
        for i = 1:j
            w = w - 2 * V(:,i) * (V(:,i)' * w);
        end
        
        % Update the rotators
        %  Determine Pj+1.
        if (j ~= length(w))
            %  Construct u for Householder reflector Pj+1.
            u = [zeros(j, 1); w(j+1:end)];
            alpha = norm(u);
            if (alpha ~= 0)
                if (w(j+1) < 0)
                    alpha = -alpha;
                end
                u(j+1) = u(j+1) + alpha;
                u = u / norm(u);
                V(:,j+1) = u;
                
                %  Apply Pj+1 to v.
                %  v = v - 2*u*(u'*v);
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
            rho = norm(w(j:j+1));
            J(:, j) = w(j:j+1) ./ rho;
            W(j+1) = - J(2, j) .* W(j);
            W(j) = conj(J(1, j)) .* W(j);
            w(j) = rho;
            w(j+1) = 0;
        end
        
        R(:, j) = w(1:restart);
        
        resid = abs(W(j+1)) / beta0;
        iter = iter + 1;
        
        % save the residual
        if (nargout > 2)
            resids(iter) = resid;
        end
        
        if (resid < tol_exit)
            break;
        end
    end
    
    % Correction
    y = R(1:j, 1:j) \ W(1:j);
    
    dx = zeros(n, 1);
    for i = j:-1:1
        dx = dx + y(i) * Z{i};
    end
    x = x + dx;
    
    if (resid < tol_exit)
        break;
    end
end

if (nargout > 2)
    resids = resids(1:iter);
end

end

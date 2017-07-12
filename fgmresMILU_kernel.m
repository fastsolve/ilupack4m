function [x, flag, iter, resids] = fgmresMILU_kernel(A, b, ...
    prec, restart, rtol, maxit, x0)
%fgmresMILU_kernel Kernel function of fgmresMILU_kernel
%   [x, flag, iter, resids] = fgmresMILU_kernel(A, b, prec, x0, options)
%
%
% See also: fgmresMILU

n = int32(size(b, 1));

% Norm of the RHS
beta0 = vec_norm2(b);
if (beta0 == 0)
    x = zeros(n, 1);
    flag = int32(0);
    iter = int32(0);
    resids = 0;
    return;
end

% Number of inner iterations
if (restart > n)
    restart = n;
end

% Maximum number of outer iterations
max_outer_iters = int32(ceil(double(maxit)/double(restart)));

% Initialize x
if (isempty(x0))
    x = zeros(n, 1);
else
    x = x0;
end

% Householder data
y = zeros(restart+1, 1);
R = zeros(restart, restart);
J = zeros(2, restart);

% Krylov subspace
V = zeros(n, restart);
% Preconditioned subspace
Z = zeros(n, restart);

if (nargout > 2)
    resids = zeros(maxit, 1);
end

iter = int32(0);
for it_outer = 1:max_outer_iters
    % Compute the initial residual
    if (it_outer > 1) || (vec_norm2(x) > 0)
        if isstruct(A)
            Ax = crs_prodAx(A, x);
        else
            Ax = A * x;
        end
        u = b - Ax;
    else
        u = b;
    end
    beta = vec_norm2(u);

    % Prepare the first Householder vector
    if (u(1) < 0)
        beta = -beta;
    end
    u(1) = u(1) + beta;
    u = u / vec_norm2(u);
    % The first Householder entry
    y(1) = - beta;
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
        v = v / vec_norm2(v);

        % Store the preconditioned vector
        Z(:,j) = ILUsol(prec, v);
        if isstruct(A)
            w = crs_prodAx(A, Z(:,j));
        else
            w = A * Z(:,j);
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
            alpha = vec_norm2(w(j+1:end));
            if (alpha ~= 0)
                if (w(j+1) < 0)
                    alpha = -alpha;
                end
                u(j+1) = u(j+1) + alpha;
                u = u / vec_norm2(u);
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
            rho = vec_norm2(w(j:j+1));
            J(:, j) = w(j:j+1) ./ rho;
            y(j+1) = - J(2, j) .* y(j);
            y(j) = conj(J(1, j)) .* y(j);
            w(j) = rho;
            w(j+1) = 0;
        end

        R(:, j) = w(1:restart);

        resid = abs(y(j+1)) / beta0;
        iter = iter + 1;

        % save the residual
        if (nargout > 2)
            resids(iter) = resid;
        end

        if (resid < rtol)
            break;
        end
    end

    % Correction
    y = backsolve(R, y, j);

    dx = zeros(n, 1);
    for i = j:-1:1
        dx = dx + y(i) * Z(:,i);
    end
    x = x + dx;

    if (resid < rtol)
        break;
    end
end

if (nargout > 2)
    resids = resids(1:iter);
end

if resid > rtol
    % Did not converge after maximum number of iterations
    flag = int32(-3);
else
    flag = int32(0);
end

end

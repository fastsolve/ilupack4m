function [L, D, U, E, F, S, p, q, flag] = ilutpCrout(A, lfil, droptol, dropstr)
% ilutpC Crout version of ILU with thresholding and pivoting
%    [L, U, E, F, S, p, q] = ilutpC(A, droptol, lfil, strategy)
%
%  === INPUT ARGUMENTS ===
%  A        : matrix in CRS or CCS format
%  lfil     : integer. The fill-in parameter. Each column of L and
%             each row of U will have a maximum of lfil elements.
%             lfil must be > 0.
%
%  droptol  : Tolerance for dropping small terms in the factorization
%
%  dropstr  : dropping strategy:
%             0 : standard dropping strategy (default) ;
%             1 : drops a term if corresponding perturbation relative
%                 to diagonal entry is small. For example for U:
%                 || L(:,i) || * |u(i,j)/D(i,i)| < tol * | D(j,j) |
%
%             2 : condition number estimator based on:
%                 rhs = (1, 1, ..., 1 )^T ;
%             3 : condition number estimator based on:
%                 rhs = ((+/-)1, ..., (+/-)1 )^T
%                 + maximizing |v_k| ;
%             4 : condition number estimator based on:
%                 rhs = ((+/-)1, ..., (+/-)1 )^T based on ;
%                 + maximizing the norm of (v_{k+1}, ..., v_n)^T );
%
%  === OUTPUT ARGUMENTS ===
%  L        : strict lower triangular part in CCS format
%  U        : strict uppper triangular part in CRS format
%  D        : diagonals
%
% Adapted from ITSOL v2.0 by Y. Saad et al.


%  DETAILS on the data structure :
%  ==============================
%
%      |  1   0  0 . . . 0 |     | U11 U12 U13 . . . U1n |
%      | L21  1  0 . . . 0 |     |  0  U22 U23 . . . U2n |
%      | L31 L32 1 . . . 0 |     |  0   0  U33 . . . U3n |
%  A = |  .   .  . .     . |  *  |  .   .   .  . . .  .  |
%      |  .   .  .   .   . |     |  .   .   .    .    .  |
%      |  .   .  .     . . |     |  .   .   .      .  .  |
%      | Ln1 Ln2 . . . . 1 |     |  0   0   0  . . . Unn |
%
%  LDU preconditioner :
%
%  D = diag( 1/U11, 1/U22, 1/U33, ..., 1/Unn )
%
%      |  0                 |    | 0 U12 U13 . . . U1n |
%      | L21  0             |    |    0  U23 . . . U2n |
%      | L31 L32 0          |    |        0  . . . U3n |
%  L = |  .   .  .  .       | U= |           . . .  .  |
%      |  .   .  .    .     |    |             .    .  |
%      |  .   .  .      .   |    |               .  .  |
%      | Ln1 Ln2 . . .  . 0 |    |                   0 |
%
%    \ - - . . . - - >
%    | \ - . . . - - >
%    | | \ - . . . - > U            L: CCS format
%    . . | \ . . . . .              U: CRS format
%    . . . . \ . . . .
%    . . . . . \ . . .
%    | | . . . . \ . .
%    | | | . . . . \ .
%    V V V . . . . . \
%      L               D


if lfil < 0
    m2c_error("ilutpCrout: Illegal value for lfil.\n");
end

NZ_DIAG = true;         % whether to replace 0 diagonal entries by small terms
DELAY_DIAG_UPD = true;  % whether to update diagonals after dropping

n = A.nrows;

%%  Workspace
%  =========
%  wL(n)      nonzero values in current column
%  Lid(n)     row numbers of nonzeros in current column
%  Lfirst(n)  At column k
%             Lfirst(1:k-1), Lfirst(j) points to next value in column j to use
%             Lfirst(k+1:n), Lfirst(j) indicates if nonzero value exists in
%                            row j of current column
%  Llist(n)   Llist(j) points to a linked list of columns that will update the
%             j-th row in U part
%
%  wU(n)      nonzero values in current row
%  Uid(n)     column numbers of nonzeros in current row
%  Ufirst(n)  At row k
%             Ufirst(1:k-1), Ufirst(j) points to next value in row j to use
%             Ufirst(k+1:n), Ufirst(j) indicates if nonzero value exists in
%                            column j of current row
%  Ulist(n)   Ulist(j) points to a linked list of rows that will update the
%             j-th column in L part

Lfirst = zeros(n, 1, 'int32');  % Zero-based indices
Llist = zeros(n, 1, 'int32');   % One-based indices
Lid = ones(n, 1, 'int32');      % One-based indices

Ufirst = zeros(n, 1, 'int32');  % Zero-based indices
Ulist = zeros(n, 1, 'int32');   % One-based indices
Uid = ones(n, 1, 'int32');      % One-based indices

wL = zeros(n, 1);
wU = zeros(n, 1);
w = zeros(n, 1);
wsym = zeros(n, 1);

L_A = ccs_tril(A);
U_A = crs_tril(A);
D = crs_diag(A);        % diagonal

% Store L as a collection of sparse columns
L = repmat(ccs_matrix(n, 1), 1, n);
% Store U as a collection of sparse rows
U = repmat(crs_matrix(1, n), n, 1);

if dropstr == 2
    eL = ones(n, 1);
    eU = ones(n, 1);
elseif dropstr == 3 || dropstr == 4
    eL = zeros(n, 1);
    eU = zeros(n, 1);
end

%% main loop
for i = 1:n
    % load column i into wL
    Lnnz = int32(0);
    tLnorm = 0.0;

    for k = L_A.col_ptr(i) : L_A.col_ptr(i+1) -1
        row = L_A.row_ind(k);
        Lfirst(row) = 1;    % zero-based
        t = L_A.val(k);
        tLnorm = tLnorm + abs(t);
        wL(row) = t;
        Lnnz = Lnnz + 1;
        Lid(Lnnz) = row;
    end

    %% load row i into wU
    Unnz = int32(0);
    tUnorm = 0.0;

    for k = U_A.row_ptr(i) : U_A.row_ptr(i+1) -1
        col = U_A.col_ind(k);
        if col ~= i          %XMJ: Why we need to check
            Ufirst(col) = 1; % zero-based
            t = U_A.val(k);
            tUnorm = tUnorm + abs(t);
            wU(col) = t;
            Unnz = Unnz + 1;
            Uid(Unnz) = col;
        end
    end

    %% update U(i) using Llist
    j = Llist(i);
    while j > 0
        lfst = Lfirst(j);
        lval = L_A.val(L_A.row_ind(j) + lfst);
        nnz_row = U_A.row_ptr(j+1) - U_A.row_ptr(j);

        for k = U_A.row_ptr(j) + Ufirst(j) : U_A.row_ptr(j+1) - 1
            col = U_A.col_ind(k);
            uval = U_A.val(k);

            if col == i
                continue;
            end
            % DIAG-OPTION: if (col == i); D[i] -= lval * uval; else */

            if Ufirst(col) == 1
                wU(col) = wU(col) - lval * uval;
            else
                % fill-in
                Ufirst(col) = 1;
                Unnz = Unnz + 1;
                Uid(Unnz) = col;
                wU(col) = -lval * uval;
            end
        end

        % update Lfirst and Llist
        lfst = lfst + 1;
        Lfirst(j) = lfst;

        if lfst < nnz_row
            newrow = L_A.row_ind(L_A.col_ptr(j) + lfst);
            iptr = j;
            j = Llist(iptr);
            Llist(iptr) = Llist(newrow);
            Llist(newrow) = iptr;
        else
            j = Llist(j);
        end
    end

    %% update L(i) using Ulist
    j = Ulist(i);

    while j > 0
        ufst = Ufirst(j);
        lfst = Lfirst(j);

        for k = L_A.col_ptr(j) + lfst : L_A.col_ptr(j+1) - 1
            row = L_A.row_ind(k);
            lval = L_A.val(k);

            if Lfirst(row) ~= 1
                % fill-in
                Lfirst(row) = int32(1);
                Lnnz = Lnnz + 1;
                Lid(Lnnz) = row;
            end
            wL(row) = wL(row) -lval * uval;
        end

        ufst = ufst + 1;
        Ufirst(j) = ufst;

        if ufst < L_A.col_ptr(j+1) - L_A.col_ptr(j)
            newcol = U_A.col_ind(U_A.row_ptr(j) + ufst);
            iptr = j;
            j = Ulist(iptr);
            Ulist(iptr) = Ulist(newcol);
            Ulist(newcol) = iptr;
        else
            j = Ulist(j);
        end
    end

    %% take care of special case when D[i] == 0
    % TODO: Need to implement diagonal pivoting
    Mnorm = (tLnorm + tUnorm) / (Lnnz+Unnz);
    if D(i) == 0
        if ~NZ_DIAG
            m2c_warning('zero diagonal encountered.\n');
            
            flag = -2;
            return;
        else
            if Mnorm == 0.0
                D(i) = 1.0;
            else
                D(i) = (1.0e-4 + droptol) * Mnorm;
            end
        end
    end

    % update diagonals [before dropping option]
    diag = abs(D(i));
    toldiag = abs(D(i)*droptol);
    D(i) = 1.0 / D(i);

    if ~DELAY_DIAG_UPD
        lu_diag = update_diagonals(lu_diag, i, ...
            wL, Lid, Lnnz, Lfirst, wU, Uid, Unnz, Ufirst);
    end
    % call different dropping funcs according to 'drop'
    % drop = 0
    if drop == 0
        std_drop(lfil, i, toldiag, toldiag, 0.0);
    elseif drop == 1
        % calculate 1-norms
        Lnorm = diag;
        for j = 1:Lnnz
            Lnorm = Lnorm + abs(wL(Lid(j)));
        end
        % compute Unorm now
        Unorm = diag;
        for j = 1:Unnz
            Unorm = Unorm + abs(wU(Uid(j)));
        end
        Lnorm = (Lnorm / (1.0 + Lnnz)) * droptol;
        Unorm = (Unorm / (1.0 + Unnz)) * droptol;
        std_drop(lfil, i, Lnorm, Unorm, 0.0);
    elseif drop == 2
        Lnorm = droptol * diag / max(1, abs(eL(i)));
        eU(i) = eU(i) * D(i);
        Unorm = droptol / max(1, abs(eU(i)));
        std_drop(lfil, i, Lnorm, Unorm, toldiag);
        
        % update eL[i+1,...,n] and eU[i+1,...,n]
        t = eL(i) * D(i);
        for j = 1:Lnnz
            row = Lid(j);
            eL(row) = eL(row) - wL(row) * t;
        end
        t = eU(i);
        for j = 1:Unnz
            col = Uid(j);
            eU(col) = eU(col) - wU(col) * t;
        end
    elseif drop == 3
        t1 = 1 - eL(i);
        t2 = -1 - eL(i);
        if abs(t1) > abs(t2)
            eL(i) = t1;
        else
            eL(i) = t2;
        end
        
        t1 = 1 - eU(i);
        t2 = -1 - eU(i);
        if fabs(t1) > fabs(t2)
            eU(i) = t1;
        else
            eU(i) = t2;
        end

        eU(i) = eU(i) * D(i);
        Lnorm = droptol * diag / max(1,abs(eL(i)));
        Unorm = droptol / max(1, abs(eU(i)));
        std_drop(lfil, i, Lnorm, Unorm, toldiag);

        % update eL[i+1,...,n] and eU[i+1,...,n]
        t = eL(i) * D(i);
        for j = 1:Lnnz
            row = Lid(j);
            eL(row) = eL(row) + wL(row) * t;
        end
        t = eU(i);
        for j = 1:Unnz
            col = Uid(j);
            eU(col) = eU(col) + wU(col) * t;
        end
    elseif drop == 4
        x1 = 1 - eL(i);
        t = x1 * D(i);
        s1 = 0;
        for j = 1:Lnnz
            row = Lid(j);
            s1 = s1 + abs(eL(row) + wL(row) * t);
        end
        
        x2 = -1 - eL(i);
        t = x2 * D(i);
        s2 = 0;
        for j = 1:Lnnz
            row = Lid(j);
            s2 = s2 + abs(eL(row) + wL(row) * t);
        end
        
        if s1 > s2
            eL(i) = x1;
        else
            eL(i) = x2;
        end
        
        Lnorm = droptol * diag / max(1, abs(eL(i)));
        x1 = (1 - eU(i)) * D(i);
        x2 = (-1 - eU(i)) * D(i);
        s1 = 0.0;
        t  = x1;
        for j = 1:Unnz
            col = Uid(j);
            s1 = s1 + abs(eU(col) + wU(col) * t);
        end
        s2 = 0.0;
        t = x2;
        for j = 1:Unnz
            col = Uid(j);
            s2 = s2 + abs(eU(col) + wU(col) * t);
        end
        if s1 > s2
            eU(i) = x1;
        else
            eU(i) = x2;
        end
        
        Unorm = droptol / max(1, abs(eU(i)));
        std_drop(lfil, i, Lnorm, Unorm, toldiag);
        % update eL[i+1,...,n] and eU[i+1,...,n]
        t = eL(i) * D(i);
        for j = 1:Lnnz
            row = Lid(j);
            eL(row) = eL(row) + wL(row) * t;
        end
        t = eU(i);
        for j = 1:Unnz
            col = Uid(j);
            eU(col) = eU(col) + wU(col) * t;
        end
    else
        m2c_error('Invalid option for dropping ...\n');
    end

    % update diagonals [after dropping option]
    if DELAY_DIAG_UPD
        lu_diag = update_diagonals(lu_diag, i, ...
            wL, Lid, Lnnz, Lfirst, wU, Uid, Unnz, Ufirst);
    end

    % reset nonzero indicators [partly reset already]
    for j = 1:Lnnz
        Lfirst(Lid(j)) = 0;
    end
    for j = 1:Unnz
        Ufirst(Uid(j)) = 0;
    end
    
    % initialize linked list for next row of U
    if ~isempty(U(i).val)
        col = Uid(1);
        Ufirst(i) = int32(0);  % Ufirst is zero-based
        Ulist(i) = Ulist(col);
        Ulist(col) = i;
    end
    
    % initialize linked list for next column of L
    if ~isempty(L(i).val)
        row = Lid(1);
        Lfirst(i) = int32(0);  % Lfirst is zero-based
        Llist(i) = Llist(row);
        Llist(row) = i;
    end
end
end


function diag = update_diagonals(diag, i, ...
    wL, Lid, Lnnz, Lfirst, wU, Uid, Unnz, Ufirst)
% update diagonals D_{i+1,...,n}

% By using the expansion arrays, only the shorter one of L(k) and U(k)
% need to be scaned, so the time complexity = O(min(Lnnz,Unnz))
scale = diag(i);

if Lnnz < Unnz
    for j = 1:Lnnz
       id = Lid(j);
       if Ufirst(id)
           diag(id) = diag(id) - wL(id) * wU(id) * scale;
       end
    end
else
    for j = 1:Unnz
        id = Uid(j);
        if Lfirst(id)
            diag(id) = diag(id) - wL(id) * wU(id) * scale;
        end
    end
end

end

%
% int comp(const void *fst, const void *snd )
% {
% /*-------------------------------------------------------------------
%  * compares two integers
%  * a callback function used by qsort
% ---------------------------------------------------------------------*/
%   int *i = (int *)fst, *j = (int *)snd;
%   if(*i > *j ) return 1;
%   if(*i < *j ) return -1;
%   return 0;
% }


function [w, Lnnz, Lfirst, Lid, Unnz, Ufirst, Uid] = std_drop(D, w, Lnnz, Lfirst, Lid, ...
    Unnz, Ufirst, Uid, lfil, i, tolL, tolU, toldiag)
% Standard Dual drop-off strategy
% ===============================
% 1) Theresholding in L and U as set by tol. Any element whose size
%    is less than some tolerance (relative to the norm of current
%    row in u or current column of L) is dropped.
% 2) Keeping only the largest lfil elements in the i-th column of L
%    and the largest lfil elements in the i-th row of U.
%    lfil    = number of elements to keep in L and U
%    tolL    = tolerance parameter used to the L factor
%    tolU    = tolerance parameter used to the U factor
%    toldiag = used for blended dropping.
% 
%    L(i,j) is dropped if | L(i,j) | < toldiag*BLEND + tolL*(1-BLEND)
%    U(i,j) is dropped if | U(i,j) | < toldiag*BLEND + tolU*(1-BLEND)

BLEND = 0.1;            % defines how to blend dropping by diagonal and
                        % other strategies. Element is always dropped when
                        % (for Lij) : Lij < B*tol*D[i]+(1-B)*Norm (inv(L)*e_k)

t = D(i);
%% drop U elements
len  = int32(0);
tolU = BLEND * toldiag + (1.0 - BLEND) * tolU;
tolL = BLEND * toldiag + (1.0 - BLEND) * tolL;

for j = 1:Unnz
    col = Uid(j);
    if abs(wU(col)) > tolU
        len = len + 1;
        Uid(len) = col;
    else
        Ufirst(col) = 0;
    end
end

%% find the largest lfil elements in row k
Unnz = len;
len = min(Unnz, lfil);

for j = 1:Unnz
    w(j) = abs(wU(Uid(j)));
end

[Unnz, len] = qsplit(w, Uid, Unnz, len);
qsort(Uid, len, comp);

% update U
if len
    U(i).row_ptr = int32([1, len + 1]);
    U(i).col_ind = zeros(len, 1, 'int32');
    U(i).val = zeros(len, 1);
end

for j = 1:len
    ipos = Uid(j);
    U(i).col_ind(j) = ipos;
    U(i).val(j) = wU(ipos);
end

for j = len:Unnz
    Ufirst(Uid(j)) = int32(0);
end
Unnz = len;

% drop L elements
len = int32(0);

for j = 1:Lnnz
    row = Lid(j);
    if abs(wL(row)) > tolL
        len = len + 1;
        Lid(len) = row;
    else
        Lfirst(row) = int32(0);
    end
end

% find the largest lfil elements in column k
Lnnz = len;
len = min(Lnnz, lfil);
for j = 1:Lnnz
    w(j) = abs(wL(Lid(j)));
end
[Lnnz, len] = qsplit(w, Lid, Lnnz, len);
qsort(Lid, len, comp);

% update L
if len
    L(i).col_ptr = int32([1, len + 1]);
    L(i).row_ind = zeros(len, 1, 'int32');
    L(i).val = zeros(len, 1);
end

for j = 1:len
    ipos = Lid(j);
    L(i).row_ind(j) = ipos;
    L(i).val(j) = wL(ipos) * t;
end

for j = len:Lnnz
    Lfirst(Lid(j)) = int32(0);
end

Lnnz = len;
end

function [L,D,U,P,Q,Dl,Dr]=AMGextractilu(PREC)
% [L,D,U,P,Q,Dl,Dr]=AMGextractilu(PREC)
%
% extract single-level ILU from ILUPACK's multilevel ILU
% such that L*D*U ~ P^T*Dl*A*Dr*Q, where A refers to the
% original matrix from which the multilevel ILU was originally 
% constructed
% 
%
% Input
% -----
% PREC        ILUPACK multilevel ILU
%
%
% Output
% ------
% L          lower unit triangular matrix
% D          (block) diagonal matrix
% U          upper unit triangular matrix
% P,Q        left and right permutation matrices
% Dl,Dr      left and right diagonal scaling matrices


coarsereduce=0;
if length(PREC)>1
   [mL,nL]=size(PREC(1).L);
   if mL==nL
      coarsereduce=1;
   end
end

% number of levels
nlev=length(PREC);
n=PREC(1).n;

% SPD-case
if PREC(1).isdefinite && PREC(1).ishermitian
   % now compute global P^TDADP ~ LDL^TD factorization
   p=1:n;
   d=ones(1,n);
   L=speye(n);
   D=speye(n);
   block=ones(1,nlev+1);
   sumnB=0;
   for lev=1:nlev
       nB=PREC(lev).nB;

       p_local=PREC(lev).p;
       rowscal=PREC(lev).rowscal;
       
       L_loc=PREC(lev).L;
       D_loc=PREC(lev).D;
       [mL,nL]=size(L_loc);

       % data structure yields L*iD*L' where diag(L)=iD^{-1}
       L_loc=L_loc*D_loc;
       D_loc=inv(D_loc);D_loc=(D_loc+D_loc')./2;
       % now we have L*D*L', such that L has unit diagonal part

       % (1,1) block
       D(sumnB+1:sumnB+nB,sumnB+1:sumnB+nB)=D_loc;

       if mL==nL
	  L(sumnB+1:sumnB+nB,sumnB+1:sumnB+nB)=tril(L_loc);
    
	  % (2,1) block only exists for lev<nlev
	  if lev<nlev
	     L(sumnB+nB+1:end,sumnB+1:sumnB+nB)=(PREC(lev).E/L_loc')/D_loc;
	  end
       else
	  L(sumnB+1:end,sumnB+1:sumnB+nB)=tril(L_loc);
       end
    
       % scaling and permuting previous parts of L only necessary if lev>1
       if lev>1
	  L(sumnB+1:end,1:sumnB)=spdiags(rowscal',0,n-sumnB,n-sumnB)*L(sumnB+1:end,1:sumnB);
	  L(sumnB+1:end,1:sumnB)=L(sumnB+p_local,1:sumnB);
       end % if
    
       if lev==1
	  d=rowscal;
	  p=p_local;
       else
	  % enlarge rowscale and reorder
	  for l=lev-1:-1:1
	      rowscal=[ones(1,PREC(l).nB) rowscal];
	      rowscal=rowscal(PREC(l).invq);
	  end % for l
	  d=d.*rowscal;

	  % update permutation
	  p(sumnB+1:n)=p(sumnB+p_local);
       end % if lev<nlev
    
    
       sumnB=sumnB+nB;
       block(lev+1)=sumnB+1;
   end % for i
   Dl=spdiags(d',0,n,n); 
   
   P=speye(n); P=P(:,p);

   U=L';
   Dr=Dl;
   Q=P;
elseif PREC(1).ishermitian || PREC(1).issymmetric
   % now compute global P^TDADP ~ LDL^TD factorization
   p=1:n;
   d=ones(1,n);
   L=speye(n);
   D=speye(n);
   block=ones(1,nlev+1);
   sumnB=0;
   for lev=1:nlev
       nB=PREC(lev).nB;

       p_local=PREC(lev).p;
       rowscal=PREC(lev).rowscal;
       
       L_loc=PREC(lev).L;
       D_loc=PREC(lev).D;
       [mL,nL]=size(L_loc);

       % if issparse(L_loc)
       % data structure yields L*D^{-1}*L' where diag(L)=D^{-1}
       L_loc=L_loc/D_loc;
       % now we have L*D*L', such that L has unit diagonal part
       % end
	  
       % (1,1) block
       D(sumnB+1:sumnB+nB,sumnB+1:sumnB+nB)=D_loc;

       if mL==nL
	  L(sumnB+1:sumnB+nB,sumnB+1:sumnB+nB)=tril(L_loc);
    
	  % (2,1) block only exists for lev<nlev
	  if lev<nlev
	     if PREC(1).ishermitian
	        L(sumnB+nB+1:end,sumnB+1:sumnB+nB)=(PREC(lev).E/L_loc')/D_loc;
	     else
	        L(sumnB+nB+1:end,sumnB+1:sumnB+nB)=(PREC(lev).E/L_loc.')/D_loc;
	     end
	  end
       else
	  L(sumnB+1:end,sumnB+1:sumnB+nB)=tril(L_loc);
       end
    
       % scaling and permuting previous parts of L only necessary if lev>1
       if lev>1
	  L(sumnB+1:end,1:sumnB)=spdiags(rowscal',0,n-sumnB,n-sumnB)*L(sumnB+1:end,1:sumnB);
	  L(sumnB+1:end,1:sumnB)=L(sumnB+p_local,1:sumnB);
       end % if
    
       if lev==1
	  d=rowscal;
	  p=p_local;
       else
	  % enlarge rowscale and reorder
	  for l=lev-1:-1:1
	      rowscal=[ones(1,PREC(l).nB) rowscal];
	      rowscal=rowscal(PREC(l).invq);
	  end % for l
	  d=d.*rowscal;

	  % update permutation
	  p(sumnB+1:n)=p(sumnB+p_local);
       end % if lev<nlev
    
    
       sumnB=sumnB+nB;
       block(lev+1)=sumnB+1;
   end % for i
   Dl=spdiags(d',0,n,n); 
   
   P=speye(n); P=P(:,p);

   if PREC(1).ishermitian
      D=(D+D')./2;
      U=L';
   else
      D=(D+D.')./2;
      U=L.';
   end
   Dr=Dl;
   Q=P;
else % general case
   % now compute global P^T Dl A Dr Q ~ L D U factorization
   p=1:n;
   q=1:n;
   dr=ones(1,n);
   dc=ones(1,n);
   L=speye(n);
   U=speye(n);
   D=speye(n);
   block=ones(1,nlev+1);
   sumnB=0;
   for lev=1:nlev
       nB=PREC(lev).nB;

       p_local=PREC(lev).p;
       invq_local=PREC(lev).invq;
       q_local=p_local;
       q_local(invq_local)=1:length(invq_local);
       % [min(p_local) max(p_local)]
       % [min(q_local) max(q_local)]
       rowscal=PREC(lev).rowscal;
       colscal=PREC(lev).colscal;
       
       L_loc=PREC(lev).L;
       D_loc=PREC(lev).D;
       U_loc=PREC(lev).U;
       [mL,nL]=size(L_loc);

       if issparse(L_loc)
	  % data structure yields L*D^{-1}*U where diag(L)=diag(U)=D^{-1}
	  L_loc=L_loc/D_loc;
	  U_loc=D_loc\U_loc;
	  % now we have L*D*U, such that L,U have unit diagonal part
       end

       % (1,1) block
       D(sumnB+1:sumnB+nB,sumnB+1:sumnB+nB)=D_loc;

       if mL==nL
	  L(sumnB+1:sumnB+nB,sumnB+1:sumnB+nB)=tril(L_loc);
	  U(sumnB+1:sumnB+nB,sumnB+1:sumnB+nB)=triu(U_loc);
    
	  % (2,1) (1,2) block only exist for lev<nlev
	  if lev<nlev
	     L(sumnB+nB+1:end,sumnB+1:sumnB+nB)=(PREC(lev).E/U_loc)/D_loc;
	     U(sumnB+1:sumnB+nB,sumnB+nB+1:end)=D_loc\(L_loc\PREC(lev).F);
	  end
       else
	  L(sumnB+1:end,sumnB+1:sumnB+nB)=tril(L_loc);
	  U(sumnB+1:sumnB+nB,sumnB+1:end)=triu(U_loc);
       end
    
       % scaling and permuting previous parts of L only necessary if lev>1
       if lev>1
	  % [min(sumnB+1:n),max(sumnB+1:n)]
	  % [min(sumnB+p_local),max(sumnB+p_local)]
	  L(sumnB+1:end,1:sumnB)=spdiags(rowscal',0,n-sumnB,n-sumnB)*L(sumnB+1:end,1:sumnB);
	  L(sumnB+1:end,1:sumnB)=L(sumnB+p_local,1:sumnB);

	  U(1:sumnB,sumnB+1:end)=U(1:sumnB,sumnB+1:end)*spdiags(colscal',0,n-sumnB,n-sumnB);
	  % [min(sumnB+1:n),max(sumnB+1:n)]
	  % [min(sumnB+q_local),max(sumnB+q_local)]
	  U(1:sumnB,sumnB+1:end)=U(1:sumnB,sumnB+q_local);
       end % if
    
       if lev==1
	  dr=rowscal;
	  dc=colscal;
	  p=p_local;
	  q=q_local;
       else
	  % enlarge rowscale and reorder
	  for l=lev-1:-1:1
	      rowscal=[ones(1,PREC(l).nB) rowscal];
	      rowscal(PREC(l).p)=rowscal;
	      colscal=[ones(1,PREC(l).nB) colscal];
	      colscal=colscal(PREC(l).invq);
	  end % for l
	  dr=dr.*rowscal;
	  dc=dc.*colscal;

	  % update permutation
	  p(sumnB+1:n)=p(sumnB+p_local);
	  q(sumnB+1:n)=q(sumnB+q_local);
       end % if lev<nlev
    
    
       sumnB=sumnB+nB;
       block(lev+1)=sumnB+1;
   end % for i
   Dl=spdiags(dr',0,n,n); 
   Dr=spdiags(dc',0,n,n); 
   
   P=speye(n); P=P(:,p);
   Q=speye(n); Q=Q(:,q);
end % if-else



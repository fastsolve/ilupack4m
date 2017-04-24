function [L,D,U,P,Q,Dl,Dr]=AMGextractbilu(PREC,threshold,maxsize)
% [BL,BD,BU,P,Q,Dl,Dr]=AMGextractbilu(PREC,threshold,maxsize)
%
% extract single-level block ILU from ILUPACK's multilevel ILU
% such that BL*BD*BU ~ P^T*Dl*A*Dr*Q, where A refers to the
% original matrix from which the multilevel ILU was originally 
% constructed
% 
%
% Input
% -----
% PREC        ILUPACK multilevel ILU
% threshold   subsequent sparse columns of L / rows of U are merged if they
%             share a given percentage of common nonzeros as defined by
%             threshold in (0,1], threshold==1 refers to 100% coincidence
% maxsize     optionally defines the maximum block size, except for the
%             (incomplete) LU decomposition on the final level. If this is
%             dense (which happens quite often), then it is returned as a
%             single dense block even if its size exceed "maxsize"
%
%
% Output
% ------
% BL         block lower unit triangular matrix stored as cell structure
% BD         (block) diagonal matrix stored as cell structure
% BU         block upper unit triangular matrix stored as cell structure
% P,Q        left and right permutation matrices
% Dl,Dr      left and right diagonal scaling matrices

% tolerance for rounding errors in L/U
tol=10*eps;
n=size(PREC(1).n,1);
if nargin==2
   maxsize=n+1;
end % end if


coarsereduce=0;
if length(PREC)>1
   [mL,nL]=size(PREC(1).L);
   if mL==nL
      coarsereduce=1;
   end
end
kblock=1;

% number of levels
nlev=length(PREC);
n=PREC(1).n;

% SPD-case
if PREC(1).isdefinite && PREC(1).ishermitian
   % now compute global P^TDADP ~ LDL^TD factorization
   block=ones(1,nlev+1);
   sumnB=0;
   levstart=zeros(1,nlev+1);
   levstart(1)=1;
   for lev=1:nlev
       nB=PREC(lev).nB;
      
       L_loc=PREC(lev).L;
       D_loc=PREC(lev).D;
       [mL,nL]=size(L_loc);

       % data structure yields L*iD*L' where diag(L)=iD^{-1}
       L_loc=L_loc*D_loc;
       D_loc=inv(D_loc);
       D_loc=(D_loc+D_loc')./2;
       % now we have L*D*L', such that L has unit diagonal part

       % (2,1) block is not stored explicitly?
       if mL==nL && lev<nlev
	  L_loc=[L_loc; ...
	         (PREC(lev).E/L_loc')/D_loc];
       end % if
       
       % merge columns of a sparse L_loc to block columns
       if issparse(L_loc)
	  i=1;
	  while i<=nB
                % strict lower triangular part in column i
		I=find(L_loc(:,i));
		mx=max(abs(L_loc(I,i)));
		I=I(find(abs(L_loc(I,i))>=tol*mx));
		I=setdiff(I,i);
		j=i+1;
		flag=-1;
		while j<=nB && flag
		      % strict lower triangular part in column j
		      J=find(L_loc(:,j));
		      mx=max(abs(L_loc(J,j)));
		      J=J(find(abs(L_loc(J,j))>=tol*mx));
		      J=setdiff(J,i:j);
		      % common entries
		      IcapJ=intersect(I,J);
		      % total nonzeros excluding j
		      IcupJ=setdiff(union(I,J),j);
		      if length(IcapJ)<threshold*length(IcupJ) || j-i+1>maxsize
			 flag=0;
			 j=j-1;
		      else
			 I=IcupJ;
			 j=j+1;
		      end % if-else
	        end % while j<=nB && flag

		% unite columns i:j
		% determine final column j
		if flag
		   j=nB;
		end
		% 2x2 diagonal while truncating? 
		if j<nB 
	           if D_loc(j+1,j)~=0
		      j=j+1;
		      % For simplicity also add column j+1
		      % strict lower triangular part in column j
		      J=setdiff(find(L_loc(:,j)),j);
		      % exclude j
		      I=setdiff(I,j);
		      % total nonzeros
		      I=union(I,J);
		   end
		end

		J=i:j;
		% I_old=I;
		I=setdiff(I,J);
		% if length(I)~=length(I_old)
		%    [i j, setdiff(I_old,I)']
		% end
		LJJ=full(tril(L_loc(J,J)));
		L{kblock}.D=LJJ;
		L{kblock}.I=sumnB+I';
		L{kblock}.J=sumnB+J;
		
		D{kblock}.D=D_loc(J,J);
		D{kblock}.D=(D{kblock}.D+D{kblock}.D')./2;
		D{kblock}.J=sumnB+J;

		L{kblock}.L=full(L_loc(I,J))/LJJ;
	     
		kblock=kblock+1;
		i=j+1;
          end % while i<=nB
       else % L_loc is dense, this can happen at most at the final level
	    J=sumnB+1:sumnB+nB;
	    L{kblock}.D=L_loc;
	    L{kblock}.L=zeros(0,nB);
	    L{kblock}.I=[];
	    L{kblock}.J=J;
		
	    D{kblock}.D=D_loc;
	    D{kblock}.D=(D{kblock}.D+D{kblock}.D')./2;
	    D{kblock}.J=J;
	    kblock=kblock+1;
       end % if-else issparse(L_loc)
              
       sumnB=sumnB+nB;
       block(lev+1)=sumnB+1;
       % keep track where the next level starts inside the block factorization
       levstart(lev+1)=kblock;
   end % for lev


   % initial scaling and permutation
   rowscal=PREC(nlev).rowscal;
   p=PREC(nlev).p;
   for lev=nlev-1:-1:1
       % size of the leading block
       nB=PREC(lev).nB;
       
       % reorder and rescale sub-diagonal blocks with respect to the inner
       % (later) scalings and permutations
       invp(p)=1:length(p);       
       for k=levstart(lev):levstart(lev+1)-1
	   I=L{k}.I;
	   l=length(I);
	   if l>0 
	      sumnB=block(lev+1)-1;
	      I1=I(find(I<=sumnB));
	      I2=I(find(I>sumnB));
	      l1=length(I1);
	      l2=length(I2);
	      % only the lower part is rescaled and permuted
	      if l2>0
		 [Iperm,II]=sort(invp(I2-sumnB));
		 L{k}.I=[I1, Iperm+sumnB];
		 L{k}.L(l1+1:end,:)=spdiags(rowscal(I2-sumnB)',0,l2,l2)*L{k}.L(l1+1:end,:);
		 L{k}.L(l1+1:end,:)=L{k}.L(l1+II,:);
	      end % if l>0
	   end % if l>0
       end % for k

       % expand p from inner (later) levels
       p=[1:nB, nB+p];
       % expand scalings from inner (later) levels
       rowscal=[ones(1,nB) rowscal];
       
       % exchange scalings from inner (later) levels with the permutation from
       % the current level "lev"
       rowscal=rowscal(PREC(lev).invq);
       % current scaling
       rowscal_local=PREC(lev).rowscal;
       % accumulate scalings
       rowscal=rowscal.*rowscal_local;
       
       % current permutation
       p_local=PREC(lev).p;
       % update permutation
       p=p_local(p);
   end % for lev
   

   Dl=spdiags(rowscal',0,n,n); 
   P=speye(n); P=P(:,p);

   for k=1:length(L)
       U{k}.J=L{k}.J;
       U{k}.I=L{k}.I;
       U{k}.D=L{k}.D';
       U{k}.U=L{k}.L';
   end % for k
   Dr=Dl;
   Q=P;
elseif PREC(1).ishermitian || PREC(1).issymmetric
   % now compute global P^TDADP ~ LDL^TD factorization      
   block=ones(1,nlev+1);
   sumnB=0;
   levstart=zeros(1,nlev+1);
   levstart(1)=1;
   for lev=1:nlev
       nB=PREC(lev).nB;
      
       L_loc=PREC(lev).L;
       D_loc=PREC(lev).D;
       if PREC(1).ishermitian
          D_loc=(D_loc+D_loc')./2;
       else
          D_loc=(D_loc+D_loc.')./2;
       end
       [mL,nL]=size(L_loc);

       % data structure yields L*D^{-1}*L' where diag(L)=D^{-1}
       L_loc=L_loc/D_loc;
       % now we have L*D*L', such that L has unit diagonal part

       % (2,1) block is not stored explicitly?
       if mL==nL && lev<nlev
	  if PREC(1).ishermitian
	     L_loc=[L_loc; ...
	            (PREC(lev).E/L_loc')/D_loc];
 	  else
	     L_loc=[L_loc; ...
	            (PREC(lev).E/L_loc.')/D_loc];
	  end % if-else
       end % if
       
       % merge columns of a sparse L_loc to block columns
       if issparse(L_loc)
	  i=1;
	  while i<=nB
                % strict lower triangular part in column i
		I=find(L_loc(:,i));
		mx=max(abs(L_loc(I,i)));
		I=I(find(abs(L_loc(I,i))>=tol*mx));
		I=setdiff(I,i);
		j=i+1;
		flag=-1;
		while j<=nB && flag
		      % strict lower triangular part in column j
		      J=find(L_loc(:,j));
		      mx=max(abs(L_loc(J,j)));
		      J=J(find(abs(L_loc(J,j))>=tol*mx));
		      J=setdiff(J,i:j);
		      % common entries
		      IcapJ=intersect(I,J);
		      % total nonzeros excluding j
		      IcupJ=setdiff(union(I,J),j);
		      if length(IcapJ)<threshold*length(IcupJ) || j-i+1>maxsize
			 flag=0;
			 j=j-1;
		      else
			 I=IcupJ;
			 j=j+1;
		      end % if-else
	        end % while j<=nB && flag

		% unite columns i:j
		% determine final column j
		if flag
		   j=nB;
		end
		% 2x2 diagonal while truncating? 
		if j<nB 
	           if D_loc(j+1,j)~=0
		      j=j+1;
		      % For simplicity also add column j+1
		      % strict lower triangular part in column j
		      J=setdiff(find(L_loc(:,j)),j);
		      % exclude j
		      I=setdiff(I,j);
		      % total nonzeros
		      I=union(I,J);
		   end
		end

		J=i:j;
		% I_old=I;
		I=setdiff(I,J);
		% if length(I)~=length(I_old)
		%    [i j, setdiff(I_old,I)']
		% end
		LJJ=full(tril(L_loc(J,J)));
		L{kblock}.D=LJJ;
		L{kblock}.I=sumnB+I';
		L{kblock}.J=sumnB+J;
		
		D{kblock}.D=D_loc(J,J);
		if PREC(1).ishermitian
		   D{kblock}.D=(D{kblock}.D+D{kblock}.D')./2;
		else
		   D{kblock}.D=(D{kblock}.D+D{kblock}.D.')./2;
		end
		D{kblock}.J=sumnB+J;

		L{kblock}.L=full(L_loc(I,J))/LJJ;
	     
		kblock=kblock+1;
		i=j+1;
          end % while i<=nB
       else % L_loc is dense, this can happen at most at the final level
	    J=sumnB+1:sumnB+nB;
	    L{kblock}.D=L_loc;
	    L{kblock}.L=zeros(0,nB);
	    L{kblock}.I=[];
	    L{kblock}.J=J;
		
	    D{kblock}.D=D_loc;
	    if PREC(1).ishermitian
	       D{kblock}.D=(D{kblock}.D+D{kblock}.D')./2;
	    else
	       D{kblock}.D=(D{kblock}.D+D{kblock}.D.')./2;
	    end
	    D{kblock}.J=J;
	    kblock=kblock+1;
       end % if-else issparse(L_loc)
              
       sumnB=sumnB+nB;
       block(lev+1)=sumnB+1;
       % keep track where the next level starts inside the block factorization
       levstart(lev+1)=kblock;
   end % for lev


   % initial scaling and permutation
   rowscal=PREC(nlev).rowscal;
   p=PREC(nlev).p;
   for lev=nlev-1:-1:1
       % size of the leading block
       nB=PREC(lev).nB;
       
       % reorder and rescale sub-diagonal blocks with respect to the inner
       % (later) scalings and permutations
       invp(p)=1:length(p);       
       for k=levstart(lev):levstart(lev+1)-1
	   I=L{k}.I;
	   l=length(I);
	   if l>0 
	      sumnB=block(lev+1)-1;
	      I1=I(find(I<=sumnB));
	      I2=I(find(I>sumnB));
	      l1=length(I1);
	      l2=length(I2);
	      % only the lower part is rescaled and permuted
	      if l2>0
		 [Iperm,II]=sort(invp(I2-sumnB));
		 L{k}.I=[I1, Iperm+sumnB];
		 L{k}.L(l1+1:end,:)=spdiags(rowscal(I2-sumnB)',0,l2,l2)*L{k}.L(l1+1:end,:);
		 L{k}.L(l1+1:end,:)=L{k}.L(l1+II,:);
	      end % if l>0
	   end % if l>0
       end % for k

       % expand p from inner (later) levels
       p=[1:nB, nB+p];
       % expand scalings from inner (later) levels
       rowscal=[ones(1,nB) rowscal];
       
       % exchange scalings from inner (later) levels with the permutation from
       % the current level "lev"
       rowscal=rowscal(PREC(lev).invq);
       % current scaling
       rowscal_local=PREC(lev).rowscal;
       % accumulate scalings
       rowscal=rowscal.*rowscal_local;
       
       % current permutation
       p_local=PREC(lev).p;
       % update permutation
       p=p_local(p);
   end % for lev
   

   Dl=spdiags(rowscal',0,n,n); 
   P=speye(n); P=P(:,p);

   for k=1:length(L)
       U{k}.J=L{k}.J;
       U{k}.I=L{k}.I;
       if PREC(1).ishermitian
	  U{k}.D=L{k}.D';
	  U{k}.U=L{k}.L';
       else
	  U{k}.D=L{k}.D.';
	  U{k}.U=L{k}.L.';
       end % if-else
   end % for k
   Dr=Dl;
   Q=P;

else % general case
   % now compute global P^T Dl A Dr Q ~ L D U factorization
   block=ones(1,nlev+1);
   sumnB=0;
   levstart=zeros(1,nlev+1);
   levstart(1)=1;
   for lev=1:nlev
       nB=PREC(lev).nB;
      
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

       % (2,1) block is not stored explicitly?
       if mL==nL && lev<nlev
	  L_loc=[L_loc; ...
	         (PREC(lev).E/U_loc)/D_loc];
	  U_loc=[U_loc, D_loc\(L_loc\PREC(lev).F)];
       end % if
       
       % merge columns of a sparse L_loc to block columns
       if issparse(L_loc)
	  i=1;
	  while i<=nB
                % strict lower triangular part in column i
		IL=find(L_loc(:,i));
		mx=max(abs(L_loc(IL,i)));
		IL=IL(find(abs(L_loc(IL,i))>=tol*mx));
		IL=setdiff(IL,i);

		IU=find(U_loc(i,:));
		mx=max(abs(U_loc(i,IU)));
		IU=IU(find(abs(U_loc(i,IU))>=tol*mx));
		IU=setdiff(IU,i);
		I=union(IL,IU);
		
		j=i+1;
		flag=-1;
		while j<=nB && flag
		      % strict lower triangular part in column j
		      JL=find(L_loc(:,j));
		      mx=max(abs(L_loc(JL,j)));
		      JL=JL(find(abs(L_loc(JL,j))>=tol*mx));
		      JL=setdiff(JL,i:j);

		      % strict upper triangular part in row j
		      JU=find(U_loc(j,:));
		      mx=max(abs(U_loc(j,JU)));
		      JU=JU(find(abs(U_loc(j,JU))>=tol*mx));
		      JU=setdiff(JU,i:j);

		      J=union(JL,JU);
		      
		      % common entries
		      IcapJ=intersect(I,J);
		      % total nonzeros excluding j
		      IcupJ=setdiff(union(I,J),j);
		      if length(IcapJ)<threshold*length(IcupJ) || j-i+1>maxsize
			 flag=0;
			 j=j-1;
		      else
			 I=IcupJ;
			 j=j+1;
		      end % if-else
	        end % while j<=nB && flag

		% unite columns i:j
		% determine final column j
		if flag
		   j=nB;
		end
		% 2x2 diagonal while truncating? 
		if j<nB 
	           if D_loc(j+1,j)~=0
		      j=j+1;
		      % For simplicity also add column j+1
		      % strict lower triangular part in column j
		      JL=setdiff(find(L_loc(:,j)),j);
		      % strict lower triangular part in column j
		      JU=setdiff(find(U_loc(j,:)),j);
		      % exclude j
		      I=setdiff(I,j);
		      % total nonzeros
		      I=union(I,JL);
		      I=union(I,JU);
		   end
		end

		J=i:j;
		% I_old=I;
		I=setdiff(I,J);
		% if length(I)~=length(I_old)
		%    [i j, setdiff(I_old,I)']
		% end
		LJJ=full(tril(L_loc(J,J)));
		L{kblock}.D=LJJ;
		L{kblock}.I=sumnB+I';
		L{kblock}.J=sumnB+J;

		UJJ=full(triu(U_loc(J,J)));
		U{kblock}.D=UJJ;
		U{kblock}.I=sumnB+I';
		U{kblock}.J=sumnB+J;
		
		D{kblock}.D=D_loc(J,J);
		D{kblock}.J=sumnB+J;

		L{kblock}.L=full(L_loc(I,J))/LJJ;
		U{kblock}.U=UJJ\full(U_loc(J,I));
	     
		kblock=kblock+1;
		i=j+1;
          end % while i<=nB
       else % L_loc is dense, this can happen at most at the final level
	    J=sumnB+1:sumnB+nB;
	    L{kblock}.D=L_loc;
	    L{kblock}.L=zeros(0,nB);
	    L{kblock}.I=[];
	    L{kblock}.J=J;

	    U{kblock}.D=U_loc;
	    U{kblock}.U=zeros(0,nB);
	    U{kblock}.I=[];
	    U{kblock}.J=J;
		
	    D{kblock}.D=D_loc;
	    D{kblock}.J=J;
	    kblock=kblock+1;
       end % if-else issparse(L_loc)
              
       sumnB=sumnB+nB;
       block(lev+1)=sumnB+1;
       % keep track where the next level starts inside the block factorization
       levstart(lev+1)=kblock;
   end % for lev


   % initial scaling and permutation
   rowscal=PREC(nlev).rowscal;
   colscal=PREC(nlev).colscal;
   p=PREC(nlev).p;
   invq=PREC(nlev).invq;
   q(invq)=1:length(q);
   for lev=nlev-1:-1:1
       % size of the leading block
       nB=PREC(lev).nB;
       
       % reorder and rescale sub-diagonal blocks with respect to the inner
       % (later) scalings and permutations
       invp(p)=1:length(p);       
       invq(q)=1:length(q);       
       for k=levstart(lev):levstart(lev+1)-1
	   I=L{k}.I;
	   l=length(I);
	   if l>0 
	      sumnB=block(lev+1)-1;
	      I1=I(find(I<=sumnB));
	      I2=I(find(I>sumnB));
	      l1=length(I1);
	      l2=length(I2);
	      % only the lower part is rescaled and permuted
	      if l2>0
		 [Iperm,II]=sort(invp(I2-sumnB));
		 L{k}.I=[I1, Iperm+sumnB];
		 L{k}.L(l1+1:end,:)=spdiags(rowscal(I2-sumnB)',0,l2,l2)*L{k}.L(l1+1:end,:);
		 L{k}.L(l1+1:end,:)=L{k}.L(l1+II,:);
	      end % if l2>0
	   end % if l>0

	   I=U{k}.I;
	   l=length(I);
	   if l>0 
	      sumnB=block(lev+1)-1;
	      I1=I(find(I<=sumnB));
	      I2=I(find(I>sumnB));
	      l1=length(I1);
	      l2=length(I2);
	      % only the upper part is rescaled and permuted
	      if l2>0
		 [Iperm,II]=sort(invp(I2-sumnB));
		 U{k}.I=[I1, Iperm+sumnB];
		 U{k}.U(:,l1+1:end)=U{k}.U(:,l1+1:end)*spdiags(colscal(I2-sumnB)',0,l2,l2);
		 U{k}.U(:,l1+1:end)=U{k}.U(:,l1+II);
	      end % if l2>0
	   end % if l>0
       end % for k

       % expand p,q from inner (later) levels
       p=[1:nB, nB+p];
       q=[1:nB, nB+q];
       % expand scalings from inner (later) levels
       rowscal=[ones(1,nB) rowscal];
       colscal=[ones(1,nB) colscal];
       
       % current permutations
       p_local=PREC(lev).p;
       invp_local(p_local)=1:length(p_local);
       invq_local=PREC(lev).invq;
       q_local(invq_local)=1:length(invq_local);

       % exchange scalings from inner (later) levels with the permutation from
       % the current level "lev"
       rowscal=rowscal(invp_local);
       % current scaling
       rowscal_local=PREC(lev).rowscal;
       % accumulate scalings
       rowscal=rowscal.*rowscal_local;
       % the current level "lev"
       colscal=colscal(invq_local);
       % current scaling
       colscal_local=PREC(lev).colscal;
       % accumulate scalings
       colscal=colscal.*colscal_local;
       
       % update permutation
       p=p_local(p);
       q=q_local(q);
   end % for lev
   

   Dl=spdiags(rowscal',0,n,n); 
   Du=spdiags(colscal',0,n,n); 
   P=speye(n); P=P(:,p);
   Q=speye(n); Q=Q(:,q);

end % if-else



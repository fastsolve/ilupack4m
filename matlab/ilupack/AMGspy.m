function AMGspy(A,PREC)
% AMGspy(PREC)
%
% display multilevel preconditioner PREC
%
% AMGspy(A,PREC)
%
% display remapped original matrix A associated with the sequence of 
% reorderings given by PREC


if nargin==1
   PREC=A;

   n=PREC(1).n;

   % do we have a parallel preconditioner?
   if isfield(PREC,'ompparts')
      [hght,I]=sort(PREC.omptab.hght);
      hght=fliplr(hght); mxhght=hght(1);
      I=fliplr(I);
      p_local=cell(1,length(PREC.ompparts));
      nlevels=0;
      nz=0;
      for i=1:length(PREC.ompparts)
	  j=I(i);
	  nz=nz+AMGnnz(PREC.ompparts{j});
	  l=length(PREC.ompparts{j});
	  nlevels=nlevels+l;
	  % leaf case, extract local permutation
	  if hght(i)==mxhght
	     p=PREC.invq(PREC.ompparts{j}(1).p_local);
	  else % extract permutation from child
	     p=p_local{j};
	  end % if-else

	  % number of constraints
	  if length(PREC.ompparts{j})>0
	     n_constraints=PREC.ompparts{j}(1).ispartial;
	     n_local=length(p)-n_constraints;
	     if hght(i)==mxhght
		AMGspypartial(PREC.ompparts{j},p,n_constraints,n,1);
	     else
		AMGspypartial(PREC.ompparts{j},p,n_constraints,n,0);
	     end
	     pause(0.001)
	  else
	     n_constraints=0;
	     n_local=0;;
	  end % if-else
	  parent=PREC.omptab.tree(j);
	     
	  % if we are not the root
	  if parent
	     % size of the final level
	     nl=PREC.ompparts{j}(l).n;
	     pn=p(end-nl+1:end);
	     if isempty(p_local{parent})
		p_local{parent}=pn;
	     else
		p_local{parent}=union(p_local{parent},pn);
	     end % if-else
	     % pn, p_local{parent}
	  end % if
	  hold on

	  if length(PREC.ompparts{j})>0
	     % horizontal lines
	     if hght(i)==mxhght
		% inner part
		plot([p(1) p(n_local)],     [p(1) p(1)]-.01, '-m');
		plot([p(1) p(n_local)],     [p(1) p(1)]-.02, '-m');
		% constraint part
		for j=n_local+1:length(p)
		    plot([p(j)-.25 p(j)+.25], [p(1) p(1)]+.01, '-m');
		    plot([p(j)-.25 p(j)+.25], [p(1) p(1)]+.02, '-m');
		end
	     else
		% inner part+constraint part
		for j=1:length(p)
		    plot([p(j)-.25 p(j)+.25], [p(1) p(1)]+.01, '-m');
		    plot([p(j)-.25 p(j)+.25], [p(1) p(1)]+.02, '-m');
		end
	     end

	     if hght(i)==mxhght
		% inner part
		plot([p(1) p(n_local)],     [p(n_local) p(n_local)]+.01, '-m');
		plot([p(1) p(n_local)],     [p(n_local) p(n_local)]+.02, '-m');
		% constraint part
		for j=n_local+1:length(p)
		    plot([p(j)-.25 p(j)+.25], [p(n_local) p(n_local)]+.01, '-m');
		    plot([p(j)-.25 p(j)+.25], [p(n_local) p(n_local)]+.02, '-m');
		end
	     else
		% inner part+constraint part
		for j=1:length(p)
		    plot([p(j)-.25 p(j)+.25], [p(n_local) p(n_local)]+.01, '-m');
		    plot([p(j)-.25 p(j)+.25], [p(n_local) p(n_local)]+.02, '-m');
		end
	     end

	     if n_local<length(p)
		if hght(i)==mxhght
		   % inner part
		   plot([p(1) p(n_local)],     [p(n_local+1) p(n_local+1)]-.02, '-m');
		   plot([p(1) p(n_local)],     [p(n_local+1) p(n_local+1)]-.01, '-m');
		   % constraint part
		   for j=n_local+1:length(p)
		       plot([p(j)-.25 p(j)+.25], [p(n_local+1) p(n_local+1)]-.02, '-m');
		       plot([p(j)-.25 p(j)+.25], [p(n_local+1) p(n_local+1)]-.01, '-m');
		   end
		else
		   % inner part+constraint part
		   for j=1:length(p)
		       plot([p(j)-.25 p(j)+.25], [p(n_local+1) p(n_local+1)]-.02, '-m');
		       plot([p(j)-.25 p(j)+.25], [p(n_local+1) p(n_local+1)]-.01, '-m');
		   end
		end
	     end
	  
	     if hght(i)==mxhght
		% inner part
		plot([p(1) p(n_local)],     [p(end) p(end)]+.01, '-m');
		plot([p(1) p(n_local)],     [p(end) p(end)]+.02, '-m');
		% constraint part
		for j=n_local+1:length(p)
		   plot([p(j)-.25 p(j)+.25], [p(end) p(end)]+.01, '-m');
		   plot([p(j)-.25 p(j)+.25], [p(end) p(end)]+.02, '-m');
		end
	     else
		% inner part+constraint part
		for j=1:length(p)
		   plot([p(j)-.25 p(j)+.25], [p(end) p(end)]+.01, '-m');
		   plot([p(j)-.25 p(j)+.25], [p(end) p(end)]+.02, '-m');
		end
	     end

	     % vertical lines
	     if hght(i)==mxhght
		% inner part
		plot([p(1) p(1)]-.01,                 [p(1) p(n_local)],     '-m');
		plot([p(1) p(1)]-.02,                 [p(1) p(n_local)],     '-m');
		% constraint part
		for j=n_local+1:length(p)
		    plot([p(1) p(1)]+.01, [p(j)-.25 p(j)+.25], '-m');
		    plot([p(1) p(1)]+.02, [p(j)-.25 p(j)+.25], '-m');
		end
	     else
		% inner part+constraint part
		for j=1:length(p)
		    plot([p(1) p(1)]+.01, [p(j)-.25 p(j)+.25], '-m');
		    plot([p(1) p(1)]+.02, [p(j)-.25 p(j)+.25], '-m');
		end
	     end
	  
	     if hght(i)==mxhght
		% inner part
		plot([p(n_local) p(n_local)]+.01,     [p(1) p(n_local)],     '-m');
		plot([p(n_local) p(n_local)]+.02,     [p(1) p(n_local)],     '-m');
		% constraint part
		for j=n_local+1:length(p)
		    plot([p(n_local) p(n_local)]+.01, [p(j)-.25 p(j)+.25], '-m');
		    plot([p(n_local) p(n_local)]+.02, [p(j)-.25 p(j)+.25], '-m');
		end
	     else
		% inner part+constraint part
		for j=1:length(p)
		    plot([p(n_local) p(n_local)]+.01, [p(j)-.25 p(j)+.25], '-m');
		    plot([p(n_local) p(n_local)]+.02, [p(j)-.25 p(j)+.25], '-m');
		end
	     end
	  
	     if n_local<length(p)
		if hght(i)==mxhght
		   % inner part
		   plot([p(n_local+1) p(n_local+1)]-.02, [p(1) p(n_local)],     '-m');
		   plot([p(n_local+1) p(n_local+1)]-.01, [p(1) p(n_local)],     '-m');
		   % constraint part
		   for j=n_local+1:length(p)
		       plot([p(n_local+1) p(n_local+1)]-.02, [p(j)-.25 p(j)+.25], '-m');
		       plot([p(n_local+1) p(n_local+1)]-.01, [p(j)-.25 p(j)+.25], '-m');
		   end
		else
		   % inner part+constraint part
		   for j=1:length(p)
		       plot([p(n_local+1) p(n_local+1)]-.02, [p(j)-.25 p(j)+.25], '-m');
		       plot([p(n_local+1) p(n_local+1)]-.01, [p(j)-.25 p(j)+.25], '-m');
		   end
		end
	     end % if
	  
	     if hght(i)==mxhght
		% inner part
		plot([p(end) p(end)]+.01,             [p(1) p(n_local)],     '-m');
		plot([p(end) p(end)]+.02,             [p(1) p(n_local)],     '-m');
		% constraint part
		for j=n_local+1:length(p)
		    plot([p(end) p(end)]+.01, [p(j)-.25 p(j)+.25], '-m');
		    plot([p(end) p(end)]+.02, [p(j)-.25 p(j)+.25], '-m');
		end
	     else
		% inner part+constraint part
		for j=1:length(p)
		    plot([p(end) p(end)]+.01, [p(j)-.25 p(j)+.25], '-m');
		    plot([p(end) p(end)]+.02, [p(j)-.25 p(j)+.25], '-m');
		end
	     end
	  end % if
	  
	  pause(0.001);
      end % for i
      hold off
      title(['parallel ILUPACK preconditioner (', ...
	     num2str(nlevels), ' levels, ', num2str(length(PREC.ompparts)), ...
	     ' tasks, ',...
	     num2str(sum(PREC.omptab.chld==0)), ' leaves)'])
      xlabel(['nz=' num2str(nz)])
      return  
   end % if
   
   % do we have a partial factorization?
   ispartial=0;
   if isfield(PREC,'ispartial')
      ispartial=PREC.ispartial;
   end % if
   
   A=sparse(n,n);
   spy(A);
   hold on

   nlev=length(PREC);
   sumnB=0;
   for lev=1:nlev
      
       nB=PREC(lev).nB;
       [mL,nL]=size(PREC(lev).L);
       
       if lev<nlev
	  if (mL==nL)  
 	     A=[sparse(sumnB,n);...
	        sparse(nB,sumnB) PREC(lev).L sparse(nB,n-sumnB-nB);...
	        sparse(n-sumnB-nB,n)]; 
	     spy(A,'g');pause(0.001);
	  
	     A=[sparse(sumnB+nB,n);...
	        sparse(n-sumnB-nB,sumnB) PREC(lev).E sparse(n-sumnB-nB,n-sumnB-nB)]; 
	     spy(A,'r');pause(0.001);
	     if PREC(1).issymmetric | PREC(1).ishermitian
	        A=[sparse(sumnB,n);...
		   sparse(nB,sumnB) PREC(lev).L' sparse(nB,n-sumnB-nB);...
		   sparse(n-sumnB-nB,n)]; 
	        spy(A,'b');pause(0.001);
	     
	        A=[sparse(sumnB,n);...
		   sparse(nB,sumnB+nB) PREC(lev).E';...
		   sparse(n-sumnB-nB,n)]; 
	        spy(A,'r');pause(0.001);
	     else
	        A=[sparse(sumnB,n);...
		   sparse(nB,sumnB) PREC(lev).U sparse(nB,n-sumnB-nB);...
		   sparse(n-sumnB-nB,n)]; 
	        spy(A,'b');pause(0.001);
	     
	        A=[sparse(sumnB,n);...
		   sparse(nB,sumnB+nB) PREC(lev).E';...
		   sparse(n-sumnB-nB,n)]; 
	       spy(A,'r');pause(0.001);
	     end
	  
	     A=[sparse(sumnB,n);...
	        sparse(nB,sumnB) PREC(lev).D sparse(nB,n-sumnB-nB);...
	        sparse(n-sumnB-nB,n)]; 
	     spy(A,'k');pause(0.001);
	  else
 	     A=[sparse(sumnB,n);...
	        sparse(n-sumnB,sumnB) PREC(lev).L sparse(n-sumnB,n-sumnB-nB)]; 
	     spy(A,'g');pause(0.001);
	  
	     if PREC(1).issymmetric | PREC(1).ishermitian
	        A=[sparse(sumnB,n);...
		   sparse(nB,sumnB) PREC(lev).L';...
		   sparse(n-sumnB-nB,n)]; 
	        spy(A,'b');pause(0.001);
	     else
	        A=[sparse(sumnB,n);...
		   sparse(nB,sumnB) PREC(lev).U;...
		   sparse(n-sumnB-nB,n)]; 
	        spy(A,'b');pause(0.001);
	     end
	  
	     A=[sparse(sumnB,n);...
	        sparse(nB,sumnB) PREC(lev).D sparse(nB,n-sumnB-nB);...
	        sparse(n-sumnB-nB,n)]; 
	     spy(A,'k');pause(0.001);
	  end
       else % lev==nlev
	  if ispartial
	     % nothing done
	     pause(0.001);
	  else % ~ispartial
	     if issparse(PREC(lev).L)
		A=[sparse(sumnB,n);...
		   sparse(nB,sumnB) PREC(lev).L sparse(nB,n-sumnB-nB);...
		   sparse(n-sumnB-nB,n)]; 
		spy(A,'g');pause(0.001);
	     
		if PREC(1).issymmetric | PREC(1).ishermitian
		   A=[sparse(sumnB,n);...
		      sparse(nB,sumnB) PREC(lev).L' sparse(nB,n-sumnB-nB);...
		      sparse(n-sumnB-nB,n)]; 
		   spy(A,'b');pause(0.001);
		else
		   A=[sparse(sumnB,n);...
		      sparse(nB,sumnB) PREC(lev).U sparse(nB,n-sumnB-nB);...
		      sparse(n-sumnB-nB,n)]; 
		   spy(A,'b');pause(0.001);
		end
	     
		A=[sparse(sumnB,n);...
		   sparse(nB,sumnB) PREC(lev).D sparse(nB,n-sumnB-nB);...
		   sparse(n-sumnB-nB,n)]; 
		spy(A,'k');pause(0.001);
	     else
		A=[sparse(sumnB,n);...
		   sparse(nB,sumnB) tril(ones(nB,nB)) sparse(nB,n-sumnB-nB);...
		   sparse(n-sumnB-nB,n)]; 
		spy(A,'g');pause(0.001);
		
		A=[sparse(sumnB,n);...
		   sparse(nB,sumnB) triu(ones(nB,nB)) sparse(nB,n-sumnB-nB);...
		   sparse(n-sumnB-nB,n)]; 
		spy(A,'b');pause(0.001);
		
		A=[sparse(sumnB,n);...
		   sparse(nB,sumnB) eye(nB) sparse(nB,n-sumnB-nB);...
		   sparse(n-sumnB-nB,n)]; 
		spy(A,'k');pause(0.001);
	     end % if-else issparse(PREC(lev).L)
	  end % if-else ispartial
       end % if-else lev<nlev
       
       sumnB=sumnB+nB;
       
       clear A;
    end % for lev
    
    sumnB=0;
    for lev=1:nlev
       nB=PREC(lev).nB;
       if lev<nlev
	  plot([sumnB+1 n],         [sumnB+nB sumnB+nB], '-k');
	  plot([sumnB+nB sumnB+nB], [sumnB+1 n],         '-k');
	  pause(0.001);
       end
       
       sumnB=sumnB+nB;
    end % for lev
 
    
    title(['ILUPACK multilevel preconditioner (' num2str(nlev) ' levels)'])
    xlabel(['nz=' num2str(AMGnnz(PREC))])
    hold off;
    
    
    
else % two input arguments, display A
  
   nz=nnz(A);
   n=PREC(1).n;
   p=1:n;
   q=1:n;
   invq=PREC.invq;
   
   % do we have a parallel preconditioner?
   if isfield(PREC,'ompparts')
      [hght,I]=sort(PREC.omptab.hght);
      hght=fliplr(hght); mxhght=hght(1);
      I=fliplr(I);
      p_local=cell(1,length(PREC.ompparts));
      nlevels=0;
      q(PREC.invq)=1:n;
      A0=A(PREC.p,q);
      for i=1:length(PREC.ompparts)
	  j=I(i);
	  l=length(PREC.ompparts{j});
	  nlevels=nlevels+l;
	  % leaf case, extract local permutation
	  if hght(i)==mxhght
	     p=PREC.invq(PREC.ompparts{j}(1).p_local);
	  else % extract permutation from child
	     p=p_local{j};
	  end % if-else
	  AA=A0(p,p);
	  B=sparse(PREC(1).n,PREC(1).n);

	  % number of constraints
	  if length(PREC.ompparts{j})>0
	     n=PREC.ompparts{j}(1).n;
	     pp=1:n;
	     qq=1:n;
	     n_constraints=PREC.ompparts{j}(1).ispartial;
	     n_local=length(p)-n_constraints;

	     sumnB=0;
	     for lev=1:l
      
		 nB   =PREC.ompparts{j}(lev).nB;
		 pp   =PREC.ompparts{j}(lev).p;
		 invqq=PREC.ompparts{j}(lev).invq;
		 qq(invqq)=1:(n-sumnB);
		 qq=qq(1:n-sumnB);
		 invpp(pp)=1:(n-sumnB);
       
		 B(p,p)=[sparse(sumnB,n);...
		         sparse(n-sumnB,sumnB) AA(pp,qq)];
		 spy(B,'b'); 
		 pause(0.001)
		 AA=AA(pp(nB+1:n-sumnB),qq(nB+1:n-sumnB));
		 hold on;
		 B(p,p)=[sparse(sumnB+nB,n);...
		         sparse(n-sumnB-nB,sumnB+nB) AA];
		 spy(B,'w');
		 pause(0.001)
		   
		   
		 sumnB=sumnB+nB;
	     end % for lev
  
	      
	     sumnB=0;
	     for lev=1:l
		 nB=PREC.ompparts{j}(lev).nB;
		 if lev<l
		    if hght(i)==mxhght
		       plot([p(sumnB+1)  p(n_local)],  [p(sumnB+nB) p(sumnB+nB)], '-k');
		       plot([p(sumnB+nB) p(sumnB+nB)], [p(sumnB+1)  p(n_local)],  '-k');

		       for jj=n_local+1:n
			   plot([p(jj)-.25   p(jj)+.25],   [p(sumnB+nB) p(sumnB+nB)], '-k');
			   plot([p(sumnB+nB) p(sumnB+nB)], [p(jj)-.25   p(jj)+.25],   '-k');
		       end % for jj
		    else
		       for jj=sumnB+1:n
			   plot([p(jj)-.25   p(jj)+.25],   [p(sumnB+nB) p(sumnB+nB)], '-k');
			   plot([p(sumnB+nB) p(sumnB+nB)], [p(jj)-.25   p(jj)+.25],   '-k');
		       end % for jj
		    end
		    pause(0.001);
		 end
		 
		 sumnB=sumnB+nB;
	     end % for lev
   
	     
	     
	     pause(0.001)
	  else
	     n_constraints=0;
	     n_local=0;;
	  end % if-else
	  parent=PREC.omptab.tree(j);
	     
	  % if we are not the root
	  if parent
	     % size of the final level
	     nl=PREC.ompparts{j}(l).n;
	     pn=p(end-nl+1:end);
	     if isempty(p_local{parent})
		p_local{parent}=pn;
	     else
		p_local{parent}=union(p_local{parent},pn);
	     end % if-else
	     % pn, p_local{parent}
	  end % if
	  hold on

	  if length(PREC.ompparts{j})>0
	     % horizontal lines
	     if hght(i)==mxhght
		% inner part
		plot([p(1) p(n_local)],     [p(1) p(1)]-.01, '-m');
		plot([p(1) p(n_local)],     [p(1) p(1)]-.02, '-m');
		% constraint part
		for j=n_local+1:length(p)
		    plot([p(j)-.25 p(j)+.25], [p(1) p(1)]+.01, '-m');
		    plot([p(j)-.25 p(j)+.25], [p(1) p(1)]+.02, '-m');
		end
	     else
		% inner part+constraint part
		for j=1:length(p)
		    plot([p(j)-.25 p(j)+.25], [p(1) p(1)]+.01, '-m');
		    plot([p(j)-.25 p(j)+.25], [p(1) p(1)]+.02, '-m');
		end
	     end

	     if hght(i)==mxhght
		% inner part
		plot([p(1) p(n_local)],     [p(n_local) p(n_local)]+.01, '-m');
		plot([p(1) p(n_local)],     [p(n_local) p(n_local)]+.02, '-m');
		% constraint part
		for j=n_local+1:length(p)
		    plot([p(j)-.25 p(j)+.25], [p(n_local) p(n_local)]+.01, '-m');
		    plot([p(j)-.25 p(j)+.25], [p(n_local) p(n_local)]+.02, '-m');
		end
	     else
		% inner part+constraint part
		for j=1:length(p)
		    plot([p(j)-.25 p(j)+.25], [p(n_local) p(n_local)]+.01, '-m');
		    plot([p(j)-.25 p(j)+.25], [p(n_local) p(n_local)]+.02, '-m');
		end
	     end

	     if n_local<length(p)
		if hght(i)==mxhght
		   % inner part
		   plot([p(1) p(n_local)],     [p(n_local+1) p(n_local+1)]-.02, '-m');
		   plot([p(1) p(n_local)],     [p(n_local+1) p(n_local+1)]-.01, '-m');
		   % constraint part
		   for j=n_local+1:length(p)
		       plot([p(j)-.25 p(j)+.25], [p(n_local+1) p(n_local+1)]-.02, '-m');
		       plot([p(j)-.25 p(j)+.25], [p(n_local+1) p(n_local+1)]-.01, '-m');
		   end
		else
		   % inner part+constraint part
		   for j=1:length(p)
		       plot([p(j)-.25 p(j)+.25], [p(n_local+1) p(n_local+1)]-.02, '-m');
		       plot([p(j)-.25 p(j)+.25], [p(n_local+1) p(n_local+1)]-.01, '-m');
		   end
		end
	     end
	  
	     if hght(i)==mxhght
		% inner part
		plot([p(1) p(n_local)],     [p(end) p(end)]+.01, '-m');
		plot([p(1) p(n_local)],     [p(end) p(end)]+.02, '-m');
		% constraint part
		for j=n_local+1:length(p)
		   plot([p(j)-.25 p(j)+.25], [p(end) p(end)]+.01, '-m');
		   plot([p(j)-.25 p(j)+.25], [p(end) p(end)]+.02, '-m');
		end
	     else
		% inner part+constraint part
		for j=1:length(p)
		   plot([p(j)-.25 p(j)+.25], [p(end) p(end)]+.01, '-m');
		   plot([p(j)-.25 p(j)+.25], [p(end) p(end)]+.02, '-m');
		end
	     end

	     % vertical lines
	     if hght(i)==mxhght
		% inner part
		plot([p(1) p(1)]-.01,                 [p(1) p(n_local)],     '-m');
		plot([p(1) p(1)]-.02,                 [p(1) p(n_local)],     '-m');
		% constraint part
		for j=n_local+1:length(p)
		    plot([p(1) p(1)]+.01, [p(j)-.25 p(j)+.25], '-m');
		    plot([p(1) p(1)]+.02, [p(j)-.25 p(j)+.25], '-m');
		end
	     else
		% inner part+constraint part
		for j=1:length(p)
		    plot([p(1) p(1)]+.01, [p(j)-.25 p(j)+.25], '-m');
		    plot([p(1) p(1)]+.02, [p(j)-.25 p(j)+.25], '-m');
		end
	     end
	  
	     if hght(i)==mxhght
		% inner part
		plot([p(n_local) p(n_local)]+.01,     [p(1) p(n_local)],     '-m');
		plot([p(n_local) p(n_local)]+.02,     [p(1) p(n_local)],     '-m');
		% constraint part
		for j=n_local+1:length(p)
		    plot([p(n_local) p(n_local)]+.01, [p(j)-.25 p(j)+.25], '-m');
		    plot([p(n_local) p(n_local)]+.02, [p(j)-.25 p(j)+.25], '-m');
		end
	     else
		% inner part+constraint part
		for j=1:length(p)
		    plot([p(n_local) p(n_local)]+.01, [p(j)-.25 p(j)+.25], '-m');
		    plot([p(n_local) p(n_local)]+.02, [p(j)-.25 p(j)+.25], '-m');
		end
	     end
	  
	     if n_local<length(p)
		if hght(i)==mxhght
		   % inner part
		   plot([p(n_local+1) p(n_local+1)]-.02, [p(1) p(n_local)],     '-m');
		   plot([p(n_local+1) p(n_local+1)]-.01, [p(1) p(n_local)],     '-m');
		   % constraint part
		   for j=n_local+1:length(p)
		       plot([p(n_local+1) p(n_local+1)]-.02, [p(j)-.25 p(j)+.25], '-m');
		       plot([p(n_local+1) p(n_local+1)]-.01, [p(j)-.25 p(j)+.25], '-m');
		   end
		else
		   % inner part+constraint part
		   for j=1:length(p)
		       plot([p(n_local+1) p(n_local+1)]-.02, [p(j)-.25 p(j)+.25], '-m');
		       plot([p(n_local+1) p(n_local+1)]-.01, [p(j)-.25 p(j)+.25], '-m');
		   end
		end
	     end % if
	  
	     if hght(i)==mxhght
		% inner part
		plot([p(end) p(end)]+.01,             [p(1) p(n_local)],     '-m');
		plot([p(end) p(end)]+.02,             [p(1) p(n_local)],     '-m');
		% constraint part
		for j=n_local+1:length(p)
		    plot([p(end) p(end)]+.01, [p(j)-.25 p(j)+.25], '-m');
		    plot([p(end) p(end)]+.02, [p(j)-.25 p(j)+.25], '-m');
		end
	     else
		% inner part+constraint part
		for j=1:length(p)
		    plot([p(end) p(end)]+.01, [p(j)-.25 p(j)+.25], '-m');
		    plot([p(end) p(end)]+.02, [p(j)-.25 p(j)+.25], '-m');
		end
	     end
	  end % if
	  
	  pause(0.001);
      end % for i
      hold off
      if nargin==1
	 title(['parallel ILUPACK preconditioner (', ...
		num2str(nlevels), ' levels, ', num2str(length(PREC.ompparts)), ...
		' tasks, ',...
		num2str(sum(PREC.omptab.chld==0)), ' leaves)'])
      else
	 title(['parallel ILUPACK reordering (', ...
		num2str(nlevels), ' levels, ', num2str(length(PREC.ompparts)), ...
		' tasks, ',...
		num2str(sum(PREC.omptab.chld==0)), ' leaves)'])
      end
      xlabel(['nz=' num2str(nz)])
      return  
   else % sequentially display matrix A
      nlev=length(PREC);
      sumnB=0;
      for lev=1:nlev
      
	  nB=PREC(lev).nB;
	  p   =PREC(lev).p;
	  invq=PREC(lev).invq;
	  q(invq)=1:(n-sumnB);
	  q=q(1:n-sumnB);
       
	  spy([sparse(sumnB,n);...
	       sparse(n-sumnB,sumnB) A(p,q)],'b'); pause(0.001)
	  A=A(p(nB+1:n-sumnB),q(nB+1:n-sumnB));
	  hold on;
	  spy([sparse(sumnB+nB,n);...
	       sparse(n-sumnB-nB,sumnB+nB) A],'w');
	  pause(0.001)
       
	  % p(sumnB+1:n)=p(sumnB+PREC(lev).p);
	  % q(sumnB+PREC(lev).invq)=q(sumnB+1:n);
	  
	  sumnB=sumnB+nB;
      end % for lev
  
   
      sumnB=0;
      for lev=1:nlev
	  nB=PREC(lev).nB;
	  if lev<nlev
	     plot([sumnB+1 n],         [sumnB+nB sumnB+nB], '-k');
	     plot([sumnB+nB sumnB+nB], [sumnB+1 n],         '-k');
	     pause(0.001);
	  end
       
	  sumnB=sumnB+nB;
      end % for lev
   
      title(['ILUPACK multilevel reordering (' num2str(nlev) ' levels)']);
      xlabel(['nz=' num2str(nz)]);
      hold off;
   end % if-else isfield(PREC,'ompparts')
end % if-else nargin==1

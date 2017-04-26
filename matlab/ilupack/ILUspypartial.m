function ILUspypartial(PREC,p,nc,nn,isleaf)
% map local preconditioner of size n at global positions defined
% by the vector p of length n

n=PREC(1).n;
nl=length(p)-nc;

% do we have a partial factorization?
ispartial=0;
if isfield(PREC,'ispartial')
   ispartial=PREC.ispartial;
end % if
   
A=sparse(nn,nn);
spy(A);
hold on

nlev=length(PREC);
sumnB=0;
for lev=1:nlev
      
   nB=PREC(lev).nB;
   [mL,nL]=size(PREC(lev).L);
       
   if lev<nlev
      if (mL==nL)  
	 A(p,p)=[sparse(sumnB,n);...
	         sparse(nB,sumnB) PREC(lev).L sparse(nB,n-sumnB-nB);...
		 sparse(n-sumnB-nB,n)]; 
         spy(A,'g');pause(0.001);
	  
	 A(p,p)=[sparse(sumnB+nB,n);...
	         sparse(n-sumnB-nB,sumnB) PREC(lev).E sparse(n-sumnB-nB,n-sumnB-nB)]; 
         spy(A,'r');pause(0.001);
	 if PREC(1).issymmetric | PREC(1).ishermitian
	    A(p,p)=[sparse(sumnB,n);...
		    sparse(nB,sumnB) PREC(lev).L' sparse(nB,n-sumnB-nB);...
		    sparse(n-sumnB-nB,n)]; 
	    spy(A,'b');pause(0.001);
	     
	    A(p,p)=[sparse(sumnB,n);...
		    sparse(nB,sumnB+nB) PREC(lev).E';...
		    sparse(n-sumnB-nB,n)]; 
	    spy(A,'r');pause(0.001);
	 else
	    A(p,p)=[sparse(sumnB,n);...
		    sparse(nB,sumnB) PREC(lev).U sparse(nB,n-sumnB-nB);...
		    sparse(n-sumnB-nB,n)]; 
	    spy(A,'b');pause(0.001);
	     
	    A(p,p)=[sparse(sumnB,n);...
	            sparse(nB,sumnB+nB) PREC(lev).E';...
		    sparse(n-sumnB-nB,n)]; 
	    spy(A,'r');pause(0.001);
	 end
	  
	 A(p,p)=[sparse(sumnB,n);...
	         sparse(nB,sumnB) PREC(lev).D sparse(nB,n-sumnB-nB);...
	         sparse(n-sumnB-nB,n)]; 
         spy(A,'k');pause(0.001);
      else
	 A(p,p)=[sparse(sumnB,n);...
	       sparse(n-sumnB,sumnB) PREC(lev).L sparse(n-sumnB,n-sumnB-nB)]; 
	 spy(A,'g');pause(0.001);
	  
	 if PREC(1).issymmetric | PREC(1).ishermitian
	    A(p,p)=[sparse(sumnB,n);...
		    sparse(nB,sumnB) PREC(lev).L';...
		    sparse(n-sumnB-nB,n)]; 
	    spy(A,'b');pause(0.001);
	 else
	    A(p,p)=[sparse(sumnB,n);...
		    sparse(nB,sumnB) PREC(lev).U;...
		    sparse(n-sumnB-nB,n)]; 
	    spy(A,'b');pause(0.001);
	 end
	  
	 A(p,p)=[sparse(sumnB,n);...
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
	    A(p,p)=[sparse(sumnB,n);...
		    sparse(nB,sumnB) PREC(lev).L sparse(nB,n-sumnB-nB);...
		    sparse(n-sumnB-nB,n)]; 
	    spy(A,'g');pause(0.001);
	     
	    if PREC(1).issymmetric | PREC(1).ishermitian
	       A(p,p)=[sparse(sumnB,n);...
		       sparse(nB,sumnB) PREC(lev).L' sparse(nB,n-sumnB-nB);...
		       sparse(n-sumnB-nB,n)]; 
	       spy(A,'b');pause(0.001);
	    else
	       A(p,p)=[sparse(sumnB,n);...
		       sparse(nB,sumnB) PREC(lev).U sparse(nB,n-sumnB-nB);...
		       sparse(n-sumnB-nB,n)]; 
	       spy(A,'b');pause(0.001);
	    end
	     
	    A(p,p)=[sparse(sumnB,n);...
		    sparse(nB,sumnB) PREC(lev).D sparse(nB,n-sumnB-nB);...
		    sparse(n-sumnB-nB,n)]; 
	    spy(A,'k');pause(0.001);
	 else
	    A(p,p)=[sparse(sumnB,n);...
	            sparse(nB,sumnB) tril(ones(nB,nB)) sparse(nB,n-sumnB-nB);...
	            sparse(n-sumnB-nB,n)]; 
	    spy(A,'g');pause(0.001);
		
	    A(p,p)=[sparse(sumnB,n);...
		    sparse(nB,sumnB) triu(ones(nB,nB)) sparse(nB,n-sumnB-nB);...
		    sparse(n-sumnB-nB,n)]; 
	    spy(A,'b');pause(0.001);
		
	    A(p,p)=[sparse(sumnB,n);...
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
       % horizontal lines
       % inner part
       if isleaf
	  plot([p(sumnB+1) p(nl)], [p(sumnB+nB) p(sumnB+nB)], '-k');
       else
	  for j=sumnB+1:nl
	     plot([p(j)-.25 p(j)+.25], [p(sumnB+nB) p(sumnB+nB)], '-k');
	  end % for
       end
       % constraint part
       for j=nl+1:n
	   plot([p(j)-.25 p(j)+.25], [p(sumnB+nB) p(sumnB+nB)], '-k');
       end

       % vertical lines
       % inner part
       if isleaf
	  plot([p(sumnB+nB) p(sumnB+nB)], [p(sumnB+1) p(nl)], '-k');
       else
	  for j=sumnB+1:nl
	      plot([p(sumnB+nB) p(sumnB+nB)], [p(j)-.25 p(j)+.25], '-k');
	  end
       end
       % constraint part
       for j=nl+1:n
	   plot([p(sumnB+nB) p(sumnB+nB)], [p(j)-.25 p(j)+.25], '-k');
       end
       
       % plot([p(sumnB+1)  p(n)],         [p(sumnB+nB) p(sumnB+nB)], '-k');
       % plot([p(sumnB+nB) p(sumnB+nB)],  [p(sumnB+1)  p(n)],        '-k');
       pause(0.001);
    end
       
    sumnB=sumnB+nB;
end % for lev
 
  
hold off;

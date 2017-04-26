function nz=ILUnnz(PREC)
% nz=ILUnnz(PREC)
%
% total number of nonzeros of the multilevel ILU
% to be consistent with MATLAB, the preconditioner is
% treated as if it were unsymmetric

nz=0;

if isfield(PREC,'ompparts')
   for i=1:length(PREC.ompparts)
       nz=nz+ILUnnz(PREC.ompparts{i});
   end % for i
   return
end
% npartial=0;
% if isfield(PREC,'ispartial')
%    npartial=PREC.ispartial;
% end

nlev=length(PREC);
for lev=1:nlev

    if lev<nlev
       if PREC(1).issymmetric | PREC(1).ishermitian
	  nnzU=nnz(PREC(lev).L);
	  nnzF=nnz(PREC(lev).E);
       else
	  nnzU=nnz(PREC(lev).U);
	  nnzF=nnz(PREC(lev).F);
       end
       nz=nz+nnz(PREC(lev).L)+nnzU-nnz(PREC(lev).D)...
	    +nnz(PREC(lev).E)+nnzF;
       nz=nz+nnz(PREC(lev).A_H);
    else
       if isfield(PREC(lev),'L')
	  if issparse(PREC(lev).L)
	     if PREC(1).issymmetric | PREC(1).ishermitian
		nnzU=nnz(PREC(lev).L);
	     else
		nnzU=nnz(PREC(lev).U);
	     end
	     nz=nz+nnz(PREC(lev).L)+nnzU-nnz(PREC(lev).D);
	  else
	     nz=nz+PREC(lev).n*PREC(lev).n;
	  end
       end
    end
    if isfield(PREC(lev),'A')
       nz=nz+nnz(PREC(lev).A);
    end
end % for lev

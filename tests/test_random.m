 A = sprand(10, 10, 0.4);
 b = A * ones(10, 1);

 assert(condest(A) < 1000);
 M = MILUinit(A);

 scaledA = diag(M(1).rowscal)*A*diag(M(1).colscal);
 q(M(1).invq) = 1:10;
 scaledA = scaledA(M(1).p, q);

 assert(norm(scaledA(1:M(1).nB, 1:M(1).nB) -M(1).L * inv(M(1).D) * M(1).U) < 1.e-3);
 if length(M)==1
     fprintf(1, 'M has one structure.\n');
 elseif length(M)==2
     fprintf(1, 'M has two structures.\n');
     B = scaledA(1:M(1).nB, 1:M(1).nB);
     E = scaledA(11-M(2).n:10, 1:M(1).nB);
     F = scaledA(1:M(1).nB, 11-M(2).n:10);
     C = scaledA(11-M(2).n:10, 11-M(2).n:10);
     %assert(norm(M(1).E - E / M(1).U * M(1).D) < 1.e-3);
     %assert(norm(M(1).F - M(1).D * M(1).U \ F) < 1.e-3);
     T = [-M(1).E / M(1).L, speye(M(2).n)] * scaledA * ...
         [-M(1).U \ M(1).F; speye(M(2).n)];
     S = C - E * inv(B) * F;
     if norm(T-S) > 1.e-3
          T-S
     end
     scaledS = diag(M(2).rowscal) * S * diag(M(2).colscal);
     clear q2
     q2(M(2).invq) = 1:M(2).n;
     assert(norm(scaledS(M(2).p, q2) - M(2).L * M(2).D * M(2).U) < 1.e-3);
 end

 x = MILUprodMinvb(M, b);
 %assert(norm(x - ones(10,1)) < 1.e-12);
 M = ILUdelete(M);

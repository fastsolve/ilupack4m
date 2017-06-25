function test_aes_fem

[~, output] = system('gd-ls -p 0ByTwsK5_Tl_PdDVSWUZzUmxpMWs "*"');

files = strsplit(output, '\n');
rtol = 1.e-5;

for i=1:length(files)
    fname = strtrim(files{i});
    if ~isempty(fname)
        system(['gd-get -O -p 0ByTwsK5_Tl_PdDVSWUZzUmxpMWs ' fname]);
        s = load(fname);

        fprintf(1, 'Solving %s\n', fname);
        A = crs_2sparse(s.aes_fe3_linsys.row_ptr, ...
            s.aes_fe3_linsys.col_ind, s.aes_fe3_linsys.val);
        b = s.aes_fe3_linsys.b;
        x = gmresMILU(A, b, [], rtol);
        fprintf(1, 'Relative residual is %1.f, with tolerance %1.f\n', ...
            norm(A * x - b) / norm(b), rtol);
    end
end

end
function B = slice(A, dim, index)
% slice(A,2,index) = A(:,index,:)
% Written by Mo Chen (sth4nth@gmail.com).
sz = size(A);
sz(dim) = numel(index);
IDX = cell(1,ndims(A));
for i = 1:ndims(A)
    if i == dim
        idx = index;
    else
        idx = 1:sz(i);
    end
    shape = ones(1,ndims(A));
    shape(i) = sz(i);
    idx = reshape(idx,shape);
    shape = sz;
    shape(i) = 1;
    idx = repmat(idx,shape);
    IDX{i} = idx(:);
end
B = reshape(A(sub2ind(size(A),IDX{:})),sz);
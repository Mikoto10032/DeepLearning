function lnZ = betheEnergy(A, nodePot, edgePot, nodeBel, edgeBel)
% Compute Bethe free energy
% TBD: deal with log(0) for entropy
edgePot = reshape(edgePot,[],size(edgePot,3));
edgeBel = reshape(edgeBel,[],size(edgeBel,3));
Ex = dot(nodeBel,nodePot,1);
Exy = dot(edgeBel,edgePot,1);
Hx = -dot(nodeBel,log(nodeBel),1);
Hxy = -dot(edgeBel,log(edgeBel),1);
d = full(sum(logical(A),1));
lnZ = -sum(Ex)-sum(Exy)-sum((d-1).*Hx)+sum(Hxy);

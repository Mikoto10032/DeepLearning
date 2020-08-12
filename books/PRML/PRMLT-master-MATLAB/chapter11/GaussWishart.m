% Class for Gaussian-Wishart distribution used by Dirichlet process

classdef GaussWishart
        properties
         kappa_
         m_
         nu_
         U_
     end
     
     methods
         function obj = GaussWishart(kappa,m,nu,S)
             U = chol(S+kappa*(m*m'));
             obj.kappa_ = kappa;
             obj.m_ = m;
             obj.nu_ = nu;
             obj.U_ = U;
         end
         
         function obj = clone(obj)
         end
         
         function d = dim(obj)
             d = numel(obj.m_);
         end
         
         function obj = addData(obj, X)
             kappa0 = obj.kappa_;
             m0 = obj.m_;
             nu0 = obj.nu_;
             U0 = obj.U_;
             
             n = size(X,2);
             kappa = kappa0+n;
             m = (kappa0*m0+sum(X,2))/kappa;
             nu = nu0+n;
             U = chol(U0'*U0+X*X');

             obj.kappa_ = kappa;
             obj.m_ = m;
             obj.nu_ = nu;
             obj.U_ = U;
         end
        
         function obj = addSample(obj, x)
             kappa = obj.kappa_;
             m = obj.m_;
             nu = obj.nu_;
             U = obj.U_;
             
             kappa = kappa+1;
             m = m+(x-m)/kappa;
             nu = nu+1;
             U = cholupdate(U,x,'+');
             
             obj.kappa_ = kappa;
             obj.m_ = m;
             obj.nu_ = nu;
             obj.U_ = U;
         end
         
         function obj = delSample(obj, x)
             kappa = obj.kappa_;
             m = obj.m_;
             nu = obj.nu_;
             U = obj.U_;

             kappa = kappa-1;
             m = m-(x-m)/kappa;
             nu = nu-1;
             U = cholupdate(U,x,'-');
             
             obj.kappa_ = kappa;
             obj.m_ = m;
             obj.nu_ = nu;
             obj.U_ = U;
         end
         
         function y = logPredPdf(obj,X)
             kappa = obj.kappa_;
             m = obj.m_;
             nu = obj.nu_;
             U = obj.U_;
             
             d = size(X,1);
             v = (nu-d+1);
             U = sqrt((1+1/kappa)/v)*cholupdate(U,sqrt(kappa)*m,'-');
             
             X = bsxfun(@minus,X,m);
             Q = U'\X;
             q = dot(Q,Q,1);  % quadratic term (M distance)
             o = -log(1+q/v)*((v+d)/2);
             c = gammaln((v+d)/2)-gammaln(v/2)-(d*log(v*pi)+2*sum(log(diag(U))))/2;
             y = c+o;
         end
         
         function [mu, Sigma] = sample(obj)
%              Sample a Gaussian distribution from GaussianWishart prior
             kappa = obj.kappa_;
             m = obj.m_;
             nu = obj.nu_;
             U = obj.U_;
             
             Sigma = iwishrnd(U'*U,nu);
             mu = gaussRnd(m,Sigma/kappa);
         end
     end
end

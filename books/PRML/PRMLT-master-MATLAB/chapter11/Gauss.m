% Class for Gaussian distribution used by Dirichlet process
classdef Gauss
     properties
         n_
         mu_
         U_
     end
     
     methods
         function obj = Gauss(X)
             n = size(X,2);
             mu = mean(X,2);
             U = chol(X*X');
             
             obj.n_ = n;
             obj.mu_ = mu;
             obj.U_ = U;
         end
         
         function obj = clone(obj)
         end
        
         function obj = addSample(obj, x)
             n = obj.n_;
             mu = obj.mu_;
             U = obj.U_;
             
             n = n+1;
             mu = mu+(x-mu)/n;
             U = cholupdate(U,x,'+');
             
             obj.n_ = n;
             obj.mu_ = mu;
             obj.U_ = U;
         end
         
         function obj = delSample(obj, x)
             n = obj.n_;
             mu = obj.mu_;
             U = obj.U_;

             n = n-1;
             mu = mu-(x-mu)/n;
             U = cholupdate(U,x,'-');
             
             obj.n_ = n;
             obj.mu_ = mu;
             obj.U_ = U;
         end
         
         function y = logPdf(obj,X)
             n = obj.n_;
             mu = obj.mu_;
             U = obj.U_;
             d = size(X,1);
             
             U = cholupdate(U/sqrt(n),mu,'-');       % Sigma=X*X'/n-mu*mu'
             Q = U'\bsxfun(@minus,X,mu);
             q = dot(Q,Q,1);  % quadratic term (M distance)
             c = d*log(2*pi)+2*sum(log(diag(U)));   % normalization constant
             y = -0.5*(c+q);
         end
     end
end
function [ YMLE, b, psi, eta ] = iMLE( Y, Y0 )
 YBCK = 0.*Y0;
 YMLE = Y0;
 Nsteps = 0;

 [S, M] = size(Y);
 tol = 1 - 1./(S.^2);
 
 psi = zeros(M,1);
 eta = zeros(M,1);
 
 
 while sum(YBCK~=YMLE)>0 
  Nsteps = Nsteps+1;
  YBCK = YMLE;
  for i=1:M
   psi(i) = sum(YMLE>0 & Y(:,i)>0)./sum(YMLE>0);
   eta(i) = sum(YMLE<0 & Y(:,i)<0)./sum(YMLE<0);
  end
  
  psi = ((tol.* (2 * psi - 1)) + 1)./2;
  eta = ((tol.* (2 * eta - 1)) + 1)./2;  
  
  psi(isnan(psi)) = 0.5;
  eta(isnan(eta)) = 0.5; 
  
  b = mean(YMLE).*tol;
  YMLE = 0;
  for i=1:M
   YMLE = YMLE + ( log( (1-Y(:,i))./2 + Y(:,i).*psi(i) ) - ...
                   log( (1+Y(:,i))./2 - Y(:,i).*eta(i) ) );                   
  end
  YMLE = sign(YMLE);
 end
    YMLE = YMLE';
end
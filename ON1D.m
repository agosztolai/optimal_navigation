function [rho,err] = ON1D(S,rho0,p,T,tau,chi2)
%  
% Parameters -------------------------------------------------------------
L = p.L; dx = p.dx; saveInt = p.saveInt; p.chi2 = chi2;
p.T = T; p.tau = tau; numJKO = T/tau; 

if tau > T; rho = []; err = 1; return; end

% Computational time step depending on the size of JKO step
if     tau<=.01;            p.dt = tau/5;    
elseif tau> .01 && tau<=.1; p.dt = tau/10;      
elseif tau> .1  && tau<= 1; p.dt = tau/50;
elseif tau>  1;             p.dt = tau/100;    
end

% Initialise
rho = zeros(L/dx-1,floor((numJKO+1)/saveInt));
rho(:,1) = rho0(L/dx-1,dx);
rhoTemp  = rho(:,1);

S = S(dx:dx:L-dx)';

res = inf; k = 1; Jold = 0; 
fprintf(2,'\n tau = %3.3f T = %3.1f --> START \n',tau,T);
while (k <= numJKO)% && (res(k)>delta)
    k = k+1;
  
    [rhoTemp,Jnew,~,err] = graddes1D(@V,@dV,rhoTemp,S,p);
    
    %catch not converged solutions
    if err == 1; break; end 
    
    %register every pth output
    if rem(k-1,saveInt) == 0
        rho(:,(k-1)/saveInt+1) = rhoTemp;
    end
    
    Jnew = min(Jnew);
    res(k) = abs(Jold-Jnew);
    
    Jold = Jnew;
    
    % Plot
%     subplot(1,2,1)
%     hold off
%     plot(-L/2+dx:dx:L/2-dx,rho_temp)
%     subplot(1,2,2)
%     hold off
%     plot(-L/2+dx:dx:L/2-dx,S(-L/2+dx:dx:L/2-dx))
%     ylim([min(rho(:)) max(rho(:))])
%     title(['JKO step ',num2str(k)]);
%     pause(.001)
end

fprintf(2,'\n tau = %3.1f T = %3.1f --> DONE! (error = %1.0f)\n',tau,T);

end

% Cost function ----------------------------------------------------------
function E = V(D,chi1,chi2,regFac1,x,m,S)
    regFac2 = 1e-5;
    dx = x(2) - x(1);
    dist = abs(meshgrid(x)' - meshgrid(x));
    E = log(regFac2 + dist);
    E = D*log(regFac2 + m).*m ... %entropy
        + dx*chi1*(E*m).*m ... %coupling
        - chi2*S.*m ... %environment
        + regFac1*.5*m.^2;  %regularisation
end

% Derivative of cost function  -------------------------------------------
function dE = dV(D,chi1,chi2,regFac1,x,m,S)
    regFac2 = 1e-5;
    dx = x(2) - x(1);
    dist = abs(meshgrid(x)' - meshgrid(x));
    dE = log(regFac2 + dist);
    dE = D*log(regFac2 + m) ... %entropy 
        + dx*chi1*(dE*m) ... %coupling
        - chi2*S ... %environment
        + regFac1*m; %regularisation
end

% Gradient descent scheme ------------------------------------------------
function [mtau,J,res,err] = graddes1D(V,dV,m0,S,p)

%parameters
L = p.L; dx = p.dx; dt = p.dt; tau = p.tau; D = p.D; chi1 = p.chi1;
chi2 = p.chi2; regFac = p.regFac; maxIt = p.maxIt; desStep = p.desStep; 
deltaJKO = p.deltaJKO;

% Grid 
x = 0:dx:L; 
N = floor(tau/dt)+1;
M = length(x)-2;

% Matrices
% Difference operator
Grad = ( sparse(1:M-1,1:M-1,ones(1,M-1),M,M-1) ...
  - sparse(2:M,  1:M-1,ones(1,M-1),M,M-1) )/dx;

% Averaging
avg = .5*(sparse(1:M-1,1:M-1,ones(1,M-1),M,M-1)...
        + sparse(2:M,  1:M-1,ones(1,M-1),M,M-1));

% Initialise
v = zeros(M-1,N);
m = zeros(M,N); m(:,1) = m0;
adj = zeros(M,N); 
J = zeros(1,maxIt);
res = zeros(1,maxIt+1);


% Iterations
uNew = inf; k = 1; res(1) = inf; err = 0;
while (k < maxIt) && (res(k) > deltaJKO)
    uOld = uNew;
    
    % Compute (m^k+1)^i+1 using (v^k)^i
    for j = 1:N-1   
        bx = sparse(1:M-1,2:M,  v(:,j) <  0,M-1,M) ...
           + sparse(1:M-1,1:M-1,v(:,j) >= 0,M-1,M);
        B = dt*Grad*diag(v(:,j))*bx; 
        m(:,j+1) = (eye(M) - B)*m(:,j);    
    end
    
    if sum(sum(m<0)) ~= 0; fprintf(2,'Density is negative! tau %f',tau); err=1; break;  end
    
    uNew = m(:,N);
    adj(:,N) = -dV(D,chi1,chi2,regFac,x(2:M+1),uNew,S);
    
    % Compute adj^k+1
    for j = N-1:-1:1
        r = avg*v(:,j).^2;
        bx = sparse(1:M-1,2:M,  v(:,j+1) <  0,M-1,M) ...
           + sparse(1:M-1,1:M-1,v(:,j+1) >= 0,M-1,M);
        B = dt*Grad*diag(v(:,j))*bx;
        adj(:,j) = (eye(M) - B')*adj(:,j+1) - .5*dt*r;
    end
    
    % Update v^k to v^(k+1)
    v = (diff(adj)/dx + desStep*v)/(1 + desStep);    

    % Cost function             
    J(k) = dx*sum(V(D,chi1,chi2,regFac,x(2:M+1),m(:,N),S));
    res(k+1) = (dx*sum(abs(uNew-uOld).^2))^.5; %L2 error
    
    if k == 1 %fprintf(2,'It %i | J = %f \n',k,J(k)); 
    else %fprintf(2,'It %i | J = %f|res=%f|dJ=%f \n',k,J(k),norm(u_new-u_old),J(k)-J(k-1));
        if (J(k)-J(k-1)) > 0; fprintf(2,'dJ is positive! eps %f',tau); err = 1; break;  end
    end

    k = k+1;
    if k == maxIt-1; fprintf(2,'Max. iterations reached! tau %f',tau); err=1; break;  end
    
end
mtau = m(:,N);

end
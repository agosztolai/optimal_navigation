function [rho,err] = ON2D(S,rho0,p,T,tau,chi2)
%                          
% Parameters -------------------------------------------------------------
L = p.L; dx = p.dx; dy = p.dy; saveInt = p.saveInt; p.chi2 = chi2;
p.T = T; p.tau = tau; numJKO = T/tau; 

if tau > T; rho = []; err = 1; return; end

% Computational time step depending on the size of JKO step
if     tau<=.01;            p.dt = tau/5;    
elseif tau> .01 && tau<=.1; p.dt = tau/10;      
elseif tau> .1  && tau<= 1; p.dt = tau/50;
elseif tau>  1;             p.dt = tau/100;    
end

% Initialise
M = (floor(L/dx)-1)*(floor(L/dy)-1);
rho = zeros(M,floor((numJKO+1)/saveInt));
rho(:,1) = rho0(M,dx,dy);
rhoTemp = rho(:,1);

% Run
res = inf; k = 1; Jold = 0; 
fprintf(2,'\n tau = %3.3f T = %3.1f --> START \n',tau,T);
while (k <= numJKO)% && (res(k)>delta)
    k = k+1;
    
    [rhoTemp,Jnew,~,err] = graddes2D(@V,@dV,rhoTemp,S,p);
    
    %catch not converged solutions
    if err == 1; break; end 
    
    %register every pth output
    if rem(k-1,saveInt) == 0
        rho(:,(k-1)/saveInt+1) = rhoTemp;
    end
    
    Jnew = min(Jnew);
    res(k) = abs(Jold-Jnew);
    
    fprintf(2,'\n JKO step %i | tau %3.2f | res = %5.3E | J = %5.3E \n',...
            k-1,tau,res(k),Jnew); 
    
    Jold = Jnew;
end

fprintf(2,'\n tau = %3.1f T = %3.1f --> DONE! (error = %1.0f)\n',tau,T);

end

% Cost function ----------------------------------------------------------
function E = V(D,chi1,chi2,regFac1,X,Y,m,S)
    regFac2 = 1e-5;
    dx = X(1,2) - X(1,1); dy = Y(2,1) - Y(1,1);
    dist = sqrt( ( meshgrid(X(:))' - meshgrid(X(:)) ).^2 ... 
               + ( meshgrid(Y(:))' - meshgrid(Y(:)) ).^2 );
    E = log(regFac2 + dist);
    E = D*log(regFac2+m).*m ... %entropy
      + dx*dy*chi1*(E*m).*m ... %coupling
      - chi2*S(X,Y).*m ... %environment
      + regFac1*.5*m.^2; %regularisation
end

% Derivative of cost function  -------------------------------------------
function dE = dV(D,chi1,chi2,regFac1,X,Y,m,S)
    regFac2 = 1e-5;
    dx = X(1,2) - X(1,1); dy = Y(2,1) - Y(1,1);
    dist = sqrt( ( meshgrid(X(:))' - meshgrid(X(:)) ).^2 ... 
               + ( meshgrid(Y(:))' - meshgrid(Y(:)) ).^2 );
    dE = log(regFac2 + dist);       
    dE = D*log(regFac2 + m) ... %entropy
        + dx*dy*chi1*(dE*m) ... %coupling
        - chi2*S(X,Y) ... %%environment
        + regFac1*m; %regularisation
end

% Gradient descent scheme ------------------------------------------------
function [mtau,J,res,err] = graddes2D(V,dV,m0,S,p)

%parameters
L = p.L; dx = p.dx; dy = p.dy; dt = p.dt; tau = p.tau; D = p.D; chi1 = p.chi1;
chi2 = p.chi2; regFac = p.regFac; maxIt = p.maxIt; desStep = p.desStep; 
deltaJKO = p.deltaJKO;

% Grid 
x = -L/2:dx:L/2; Mx = length(x)-1;
y = -L/2:dy:L/2; My = length(y)-1;
[X,Y] = meshgrid(x(2:Mx),y(2:My));
M = (Mx-1)*(My-1);
t = 0:dt:tau; N = length(t); %time grid
mx = (Mx-2)*(My-1); my = (Mx-1)*(My-2); %staggered grid

% Matrices
%shift
Pn = coords(2:Mx-1, 1:My-1, Mx-1,My-1);
Ps = coords(1:Mx-2, 1:My-1, Mx-1,My-1);
Pe = coords(1:Mx-1, 2:My-1, Mx-1,My-1);
Pw = coords(1:Mx-1, 1:My-2, Mx-1,My-1);

% Averaging
tx = .5*(sparse(1:mx,Pn,ones(mx,1),mx,M) ...
        + sparse(1:mx,Ps,ones(1,mx),mx,M));
ty = .5*(sparse(1:my,Pe,ones(my,1),my,M) ... 
        + sparse(1:my,Pw,ones(1,my),my,M));

% Difference
Dx = sparse(Pn,1:mx,ones(1,mx),M,mx) - sparse(Ps,1:mx,ones(1,mx),M,mx);
Dy = sparse(Pe,1:my,ones(1,my),M,my) - sparse(Pw,1:my,ones(1,my),M,my);

% Identity
I  = sparse(1:M,1:M,ones(M,1),M,M); 
       
% Initialise
vx = zeros(mx,N);
vy = zeros(my,N);
m = zeros(M,N); m(:,1) = m0;
adj = zeros(M,N);
J = zeros(1,maxIt);
res = zeros(1,maxIt+1);

% Iterations
uNew = inf; res(1) = inf; k = 1; err = 0;
while (k < maxIt) && (res(k) > deltaJKO)
    uOld = uNew;
    
    % Compute m^k+1
    for j = 2:N
        bx = sparse(1:mx,Pn,vx(:,j-1) < 0,mx,M) ...
           + sparse(1:mx,Ps,vx(:,j-1) >= 0,mx,M);
        by = sparse(1:my,Pe,vy(:,j-1) < 0,my,M) ...
           + sparse(1:my,Pw,vy(:,j-1) >= 0,my,M);
        Bx = dt/dx*Dx*diag(vx(:,j-1))*bx;
        By = dt/dx*Dy*diag(vy(:,j-1))*by;
        m(:,j) = (I + Bx + By)*m(:,j-1);
    end

    % Check positivity
    if min(min(m))<0; fprintf(2,'Density is negative! tau %f',tau); err=1; break;  end
    
    % Check CFL
    if max(max(vx))>=.8*dx/dt || max(max(vy))>=.8*dx/dt    
        error('\t CFL condition violated !! \n'); 
    end

    uNew = m(:,N);
    adj(:,N) = -dV(D,chi1,chi2,regFac,X,Y,uNew,S);

    % Compute adj^k+1
    for j = N-1:-1:1
        bx = sparse(1:mx,Pn,vx(:,j+1) < 0,mx,M) ...
           + sparse(1:mx,Ps,vx(:,j+1) >= 0,mx,M);
        by = sparse(1:my,Pe,vy(:,j+1) < 0,my,M) ...
           + sparse(1:my,Pw,vy(:,j+1) >= 0,my,M);
        Bx = dt/dx*Dx*diag(vx(:,j+1))*bx;
        By = dt/dx*Dy*diag(vy(:,j+1))*by;
        r = tx'*vx(:,j+1).^2 + ty'*vy(:,j+1).^2;
        adj(:,j) = (I + Bx + By)'*adj(:,j+1) - .5*dt*r;
    end

    % Update v^k to v^(k+1)
    vx = (Dx'*adj/dx + desStep*vx)/(1 + desStep);
    vy = (Dy'*adj/dy + desStep*vy)/(1 + desStep);

    % Cost function             
    J(k) = dx*dy*sum(sum(V(D,chi1,chi2,regFac,X,Y,m(:,N),S)));
    res(k+1) = (dx*dy*sum(sum(abs(uNew-uOld).^2)))^.5; %L2 error
    
    if k==1 %;fprintf(2,'It %i | J = %f \n',k,J(k));
    else %fprintf(2,'It %i | J = %f|res=%f|dJ=%f \n',k,J(k),norm(uNew-uOld),J(k)-J(k-1));
        if (J(k)-J(k-1))>0; fprintf(2,'dJ is positive! tau %f',tau); err=1; break;  end
    end

    k = k+1;
    if k == maxIt-1; fprintf(2,'Max. iterations reached! tau %f',tau); err=1; break;  end
    
end
mtau = m(:,N);

end

% Get the coordinates in the array corresponding to the (i,j) element of 
% the kxl matrix, (M-1)x(M-1) in our case
function coord = coords(i,j,k,l)
    c = reshape(1:k*l,k,l);
    coord = c(i,j);
    coord = coord(:);
end
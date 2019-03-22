%
% This script implements the optimal navigation (ON) model in one or two 
% spatial dimensions as described in the article
%
% Gosztolai, A., Carrillo, J. A., Barahona, M., Collective search with 
% finite perception: transient dynamics and search efficiency (2019), 
% Frontiers in Physics, 5:153
%
% Please cite this article if you find the software useful.
% 
% The algorithm is based on sequentially solving the variational end-point 
% mean-field game obtained as the first-order optimality conditions of the 
% ON model. 
%
% #########################################################################
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% See <http://www.gnu.org/licenses/> for the GNU General Public License.
%
%##########################################################################

outfolder = '/home/ONmodel'; %output folder

%% Parameters ------------------------------------------------------------
p.L      = 100;      %domain length
T        = 50000;    %foraging time    
tau      = [ 0.25 1/3 0.5 2/3 0.75 1 1.25 2.5 5 10 50 100 500 1000 ]; %time horizon
s2       = [ 4 6 7 8 9 11 12 13]; %standard deviation
p.D      = .1;       %diffusion coefficient
chi2     = 0.2;      %advection coefficient
p.chi1   = 0;        %interaction strength
p.regFac = 0;%0.1;   %regularisation factor omega*rho^2

%solver parameters
p.dx       = 2.5;    %grid
p.dy       = 2.5;
p.maxIt    = 1e5;    %max iteration number
p.saveInt  = 1;      %register every pth output
p.deltaJKO = 1e-5;   %tolerance
p.desStep  = 500;    %gradient descent step

%% comment out as necessary for 1D or 2D
%1D
% S     = @(x,s) normpdf(x,p.L/2,s); %Gaussian
% rho0  = @(M,dx) ones(M,1);%/(dx*sum(ones(M,1))); % Initial distribution

%2D
S     = @(X,Y,s2) mvnpdf([X(:) Y(:)],[0 0],[s2 0; 0 s2]); %Gaussian
rho0 = @(M,dx,dy) ones(M,1);%/(dx*dy*sum(ones(M,1)));
% rho0 = mvnpdf([X(:) Y(:)],[0 0],[1 0; 0 1]);

%% Run -------------------------------------------------------------------
parpool(min(length(tau)*length(s2),40));
    
    % parallel loop
    parfor i = 1:length(tau)*length(s2)
        k = ceil(i/length(s2)); %tau
        j = i-(k-1)*length(s2); %s2

        %comment out as necessary for 1D or 2D
%         [rho{i},err(i)] = ON1D(@(x) S(x,s2(j)),rho0,p,T,tau(k),chi2); 
        [rho{i},err(i)] = ON2D(@(X,Y) S(X,Y,s2(j)),rho0,p,T,tau(k),chi2);
    end

    rho = reshape(rho,length(s2),length(tau));
    err = reshape(err,length(s2),length(tau));
    
	% save data   
	if exist(outfolder,'file') ~= 7; mkdir(outfolder); end
	filename = ['res',num2str(T),'.mat'];
	save([outfolder filename],'rho','err','tau','s2','T','-v7.3')   
    
delete(gcp)
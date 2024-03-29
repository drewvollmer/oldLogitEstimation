% estimate_preference_params.m
% Use observed purchases to estimate preference parameters for each quality
% After looking at the fit, use log price instead of price
% Drew Vollmer 2019-05-23

clear;
clc;


%% - There used to be code here importing and setting up the data

%%%%%%%%%%%%%%%%%%%%%%%
% Evaluate likelihood %
%%%%%%%%%%%%%%%%%%%%%%%

% % Get good starting values
% searchSet = sobolset(7);
% searchSet = searchSet(1:1000000,:);
% % Keep the first 1000 values with monotone alpha and beta properties
% searchSet = searchSet( searchSet(:,1) >= searchSet(:, 2), :);
% searchSet = searchSet( searchSet(:,2) >= searchSet(:, 3), :);
% searchSet = searchSet( searchSet(:,3) >= searchSet(:, 4), :);
% searchSet = searchSet( searchSet(:,5) <= searchSet(:, 6), :);
% searchSet = searchSet( searchSet(:,6) <= searchSet(:, 7), :);
% searchSet = searchSet(1:1000,:);

% objVals = zeros(length(searchSet(:,1)), 1);
% for i = 1:length(objVals)
%    objVals(i) =  calc_trans_loglik(searchSet(i,:)', data);
% end

% % Initial values are the best values from the Sobol search
% [min, minIdx] = min(objVals);
% initParams = searchSet(minIdx,:)';


% Evaluate likelihood of preference parameters
alphas = [3; 3; 3; 3];
betas = [0; .3; .6; .9];

calc_trans_loglik([alphas; [.3; .6; .9]], data)


%% Maximize by choice of parameters
options = optimset('Display', 'iter');
[opt_params, fval, exitflag, output, grad, hessian] = ...
    fminunc( @(params) -calc_trans_loglik(params, data), [alphas(1); [.3; .6; .9]], options)

l_alphas = opt_params(1);
l_betas = [0; opt_params(2:end)];


% Non-log solution:
% alpha = .054
% betas = [0; -.2776; .3429; 1.1349]

%% Calculate standard errors
% Invert the Hessian (don't take negative Hessian because we minimized -1*objectiveFn)
inv_hess = inv(hessian);
% Take the square root of the diagonals
stderrs = sqrt(diag(inv_hess));


%% Write parameters to CSV
dlmwrite('alphas.csv', l_alphas);
dlmwrite('betas.csv', l_betas);

dlmwrite('stderrs.csv', stderrs);

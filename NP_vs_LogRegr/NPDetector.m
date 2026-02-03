clc; clear; close all;

A        = 1.0;       % Signal amplitude
sigma    = 0.5;       % Noise standard deviation
P_FA     = 0.05;      % Target false alarm probability
trials   = 1e5;       % Monte Carlo trials


% NP Threshold
eta = sigma * sqrt(2) * erfcinv(2 * P_FA);


%%
%Monte Carlo Simulation

errors = 0;

for k = 1:trials
    
    % Randomly choose hypothesis (equal priors)
    H = rand > 0.5;
    
    if H == 0
        x = sigma * randn;      % H0
        true_label = 0;
    else
        x = A + sigma * randn;  % H1
        true_label = 1;
    end
    
    % NP Detector
    decision = (x > eta);
    
    % Count error
    if decision ~= true_label
        errors = errors + 1;
    end
end

%%
% Error Rate
P_e = errors / trials;

fprintf('NP Detector Error Rate = %.4f\n', P_e);

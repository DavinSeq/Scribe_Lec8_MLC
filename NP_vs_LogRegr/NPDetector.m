clc; clear; close all;

A        = 1.0;       % Signal amplitude    (Push Signal Amplitude towards 0 to see the performance decrease)
sigma    = 0.5;       % Noise standard deviation
P_FA     = 0.05;      % Target false alarm probability    (Increase this to see performance increase)
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
        true_label = 0;    % true_label = 0 denotes noise only
    else
        x = A + sigma * randn;  % H1
        true_label = 1;    % true_label = 1 denotes signal + noise
    end
    
    % NP Detector
    decision = (x > eta);    % If x>eta decision = 1 (signal+noise), else decision = 0 (noise only)
    
    % Count error
    if decision ~= true_label
        errors = errors + 1;
    end
end

%%
% Error Rate
P_e = errors / trials;

fprintf('NP Detector Error Rate = %.4f\n', P_e);

clc; clear; close all;

% Parameters
A       = 1;          % Signal amplitude  NOTE, you can visualize how detection performance deteriorates by pushing this value towards 0
sigma   = 0.5;        % Noise standard deviation
P_FA    = 0.01;       % Probability of false alarm

% Threshold (Neyman-Pearson)
eta = sigma * sqrt(2) * erfcinv(2 * P_FA);

% Generate test sample
H = 1;  % 0 = H0 (noise), 1 = H1 (signal + noise)

if H == 0
    x = sigma * randn;
else
    x = A + sigma * randn;
end

% Detector
if x > eta
    decision = 1;  % Decide H1
else
    decision = 0;  % Decide H0
end

fprintf('Received x = %.3f\n', x);
fprintf('Threshold eta = %.3f\n', eta);
fprintf('Decision = H%d\n', decision);

%% lfpreg.m
% Fits a sum of sinusoids to the raw LFP data using linear regression and
% computes the residuals
% 
%% Inputs:
%
%   lfpobs      : the raw LFP data
%   period      : the principle period for the sum
%   k           : the number of sines and cosines to add together for the sum
%   grid        : the sample values in period space to compute a single cycle of the fit waveform
%   beta        : the coefficients for the sum of sinusoids
%% Outputs:
%
%   res         : the residuals of the fit and raw data
%   beta        : the coefficients for the sum of sinusoids
%   f           : a single cycle of the fitted waveform
%   grid        : the sample values in period space to compute a single cycle of the fit waveform

function [res,beta,f,grid] = lfpreg(lfpobs,period,k,grid,beta)

if ~isempty(lfpobs)
    
    if nargin > 4, error('beta cannot be provided for nonempty lfpobs'); end
    
    n = size(lfpobs,1);
    
    t = lfpobs(:,1)*(2*pi/period);
    
    X = ones(n,2*k+1);
    
    for j = 1:k
        jt = j*t;
        X(:,2*j) = sin(jt);
        X(:,2*j+1) = cos(jt);
    end
    
    Y = lfpobs(:,2);
    
    beta = (X.'*X)\X.'*Y;
    
    res = Y-X*beta;
    
else
    
    if nargin < 5
        error('beta must be provided for empty lfpobs')
    end
    res = [];
    
    if isempty(k), k = (numel(beta)-1)/2; end
    
    if k ~= (numel(beta)-1)/2, error('k and beta do not agree'); end
    
end

if nargout > 2
    
    if nargin < 4 || isempty(grid)
        grid = (0:10^3)*(period/10^3);
    end
    
    f = repmat(beta(1),size(grid));
    
    gridp = grid*(2*pi/period);
    
    for j = 1:k
        jgrid = j*gridp;
        f = f + beta(2*j)*sin(jgrid) + beta(2*j+1)*cos(jgrid);
    end
    
end

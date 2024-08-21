%% PeriodicFilter.m
% Produces a linear filter to remove stimulation artifact by averaging
% points at a similar location in the artifact waveform
%
%% Inputs:
%
%   Period          : the period of stimulation in samples
%   TemporalWidth   : the half width of the linear filter in samples
%   PeriodWidth     : the half window size in samples in period space to average points at a similar location in the waveform
%   OmitWidth       : the number of samples to ignore on either side of the center of the window
%   Direction       : 'both', 'past', or 'future' indicating which samples to use for removal relative to the current sample
%% Outputs:
%
%   f               : a linear filter for removing stimulation artifact
%   x               : the positions in samples relative to the center

function [f,x,TemporalWidth,PeriodWidth,OmitWidth,Direction] = PeriodicFilter(Period,TemporalWidth,PeriodWidth,OmitWidth,Direction)

if nargin < 5 || isempty(Direction)
    Direction = 'both';
end
if ~any(strcmpi(Direction,{'past','future','both'}))
    error('Direction must be one of {past, future, both}')
end

if nargin < 4 || isempty(OmitWidth)
    OmitWidth = 0;
end
if OmitWidth < 0, error('OmitWidth must be nonnegative'), end

if nargin < 3 || isempty(PeriodWidth)
    PeriodWidth = Period/50;
end
if PeriodWidth > Period, error('PeriodWidth should be <= Period'), end

if nargin < 2 || isempty(TemporalWidth)
    TemporalWidth = OmitWidth;
    j = 0;
    while j < 50 && TemporalWidth < 10^6
        TemporalWidth = TemporalWidth + 1;
        j = j + (mod(TemporalWidth,Period) <= PeriodWidth || mod(TemporalWidth,Period) >= Period-PeriodWidth);
    end
end
if TemporalWidth <= OmitWidth, error('TemporalWidth must be greater than OmitWidth'); end


x = -TemporalWidth:TemporalWidth;
y = mod(x,Period);
% Find the samples relative to the center that are within PeriodWidth
f = double((y <= PeriodWidth | y >= Period-PeriodWidth) & abs(x) > OmitWidth);
% Ignore samples depending on the value of Direction
if strcmpi(Direction,'past')
    f(x>=0)=0;
elseif strcmpi(Direction,'future')
    f(x<=0)=0;
end
% Weight all valid samples to average equally 
f = -f / max(sum(f),eps(0));
f(x==0) = 1;


    
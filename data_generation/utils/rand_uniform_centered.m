function out = rand_uniform_centered(center, half_range, output_size)
% rand_uniform_centered - Generates uniform random values centered around a value.
%
% This function generates random values from a uniform distribution over the range
% [center - half_range, center + half_range], with the specified output size.
%
% Inputs:
%   center       - Center of the distribution (scalar)
%   half_range   - Half-width of the uniform distribution (scalar)
%   output_size  - Output size (scalar or vector), e.g., 10 or [5, 3]
%
% Output:
%   out - Random values uniformly distributed within [center - half_range, center + half_range]
%
% Examples:
%   rand_uniform_centered(5, 2, 10)         % 10x1 vector in range [3, 7]
%   rand_uniform_centered(0, 1, [3, 3])     % 3x3 matrix in range [-1, 1]
%
% Author: Jingyi Wu, 2025
    range_min = center - half_range;
    range_max = center + half_range;
    out = (range_max - range_min) * rand(output_size) + range_min;
end

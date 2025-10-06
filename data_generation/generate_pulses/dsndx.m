function dfdx = dsndx(x0, location, scale, shape)
    % This function calculates the derivative of the skewed normal distribution
    %
    % Usage:
    %   dfdx = dsndx(x0, location, scale, shape)
    %
    % Inputs:
    %   x0       - Input variable (scalar or vector)
    %   location - Location parameter (mean-like shift)
    %   scale    - Scale parameter (spread, similar to standard deviation)
    %   shape    - Shape parameter (controls skewness)
    %
    % Output:
    %   dfdx     - Value(s) of the derivative of the skewed normal PDF
    %
    % Notes:
    %   - The skewed normal PDF is defined by location, scale, and shape.
    %   - This function returns the derivative with respect to x0.
    %
    % Author: Jingyi Wu, 2025

    x = (x0 - location) ./ scale;
    dfdx = (2 / scale^2) .* normpdf(x) .* ...
           (-x .* normcdf(shape * x) + shape * normpdf(shape * x));
end

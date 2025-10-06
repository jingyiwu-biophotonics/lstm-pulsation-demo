function out = perlin_1d(x1, dx)
    % This function generates 1D Perlin noise that can make baseline
    % changes in the signal more realistic.
    % It is based on Ken Perlin's improved noise algorithm:
    % https://cs.nyu.edu/~perlin/noise/
    % Example use: out = perlin_1d(3, 1/200);
    %
    % Author: Jingyi Wu, 2025

    % x1 determines the range of x values (number of turns).
    % dx determines the smoothness (resolution) of the noise.

    % Generate coordinates
    x0 = 0;
    x = x0:dx:x1-dx;
    
    % Calculate number of grid points
    numx = x1 - x0;
    
    % Generate random gradient values for each grid point (between -1 and 1)
    grad = rand(1, numx+1) * 2 - 1;
    
    % Preallocate the output array
    zmat = zeros(1, length(x));
    
    % Iterate over each x value and compute the noise
    for idx_x = 1:length(x)
        % Find the current integer position (grid point)
        xi = floor(x(idx_x));
        
        % Calculate the distance from the current grid point
        t = x(idx_x) - xi;
        
        % Compute gradients at the surrounding grid points
        g0 = grad(xi+1);  % Gradient at left point
        g1 = grad(xi+2);  % Gradient at right point
        
        % Compute noise contribution from each gradient
        n0 = g0 * (t);       % Influence of left gradient
        n1 = g1 * (t - 1);   % Influence of right gradient
        
        % Use fade function to interpolate between the two contributions
        zmat(idx_x) = lerp(n0, n1, fade(t));
    end
    
    % Normalize the output to be within the range [-1, 1]
    out = zmat / max(abs(zmat));

end

% Fade function (smoothing)
function f = fade(t)
    f = t * t * t * (t * (t * 6 - 15) + 10);
end

% Linear interpolation function
function u = lerp(a, b, t)
    u = a + t * (b - a);
end

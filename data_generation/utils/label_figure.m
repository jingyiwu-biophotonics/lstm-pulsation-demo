function label_figure(label, x_pos, y_pos, font_size, font_weight)
% label_figure - Adds a normalized-position label (e.g., (a), (b), etc.) to the current figure.
%
% Useful for labeling subplots or figure panels. The label is placed using normalized units.
%
% Inputs:
%   label        - Character or string label to display (e.g., 'a', 'b', '1')
%   x_pos        - Horizontal position in normalized units (default = 0)
%   y_pos        - Vertical position in normalized units (default = 1)
%   font_size    - Font size of the label (default = 15)
%   font_weight  - Font weight: 'normal', 'bold', etc. (default = 'normal')
%
% Example:
%   label_figure('a')                            % Adds (a) at top-left
%   label_figure('b', 0.1, 0.95, 12, 'bold')     % Adds (b) at custom position and style
%
% Author: Jingyi Wu, 2025
    if nargin < 2 || isempty(x_pos)
        x_pos = 0;
    end
    if nargin < 3 || isempty(y_pos)
        y_pos = 1;
    end
    if nargin < 4 || isempty(font_size)
        font_size = 12;
    end
    if nargin < 5 || isempty(font_weight)
        font_weight = 'bold';
    end

    text(x_pos, y_pos, ['(', label, ')'], ...
        'Units', 'normalized', ...
        'VerticalAlignment', 'top', ...
        'FontSize', font_size, ...
        'FontWeight', font_weight);
end

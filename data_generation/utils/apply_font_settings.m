function apply_font_settings(fontSize, fontWeight, fontName, lineWidth)
    % APPLY_FONT_SETTINGS Applies font settings to all text elements in the figure.
    %
    % Usage:
    %   apply_font_settings() % Uses default settings
    %   apply_font_settings(15, 'bold', 'Arial', 1.5) % Custom settings
    %
    % Inputs (optional):
    %   fontSize   - Numeric value for font size (default: 13.5)
    %   fontWeight - Font weight ('normal' or 'bold', default: 'bold')
    %   fontName   - String specifying font name (default: 'Arial')
    %   lineWidth  - Numeric value for axes line width (default: 1.2)
    %
    % Author: Jingyi Wu, 2025
    
    % Set default values if not provided
    if nargin < 1 || isempty(fontSize), fontSize = 12; end
    if nargin < 2 || isempty(fontWeight), fontWeight = 'bold'; end
    if nargin < 3 || isempty(fontName), fontName = 'Arial'; end
    if nargin < 4 || isempty(lineWidth), lineWidth = 1.2; end

    % Apply settings to all axes in the current figure
    all_axes = findall(gcf, 'Type', 'axes');
    for ax = all_axes'
        set(ax, 'FontSize', fontSize, 'FontWeight', fontWeight, ...
            'FontName', fontName, 'LineWidth', lineWidth);
    end

    % Apply settings to legend (if it exists)
    lgd = findobj(gcf, 'Type', 'legend');
    if ~isempty(lgd)
        set(lgd, 'FontSize', fontSize, 'FontWeight', fontWeight, 'FontName', fontName);
    end

    % Apply settings to sgtitle (if it exists)
    sgt = findobj(gcf, 'Type', 'text', '-and', 'Tag', 'sgtitle');
    if ~isempty(sgt)
        set(sgt, 'FontSize', fontSize, 'FontWeight', fontWeight, 'FontName', fontName);
    end
end

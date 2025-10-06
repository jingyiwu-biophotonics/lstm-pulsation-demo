function set_style(fontSize, lineWidth)
    % Helper function for figure style.
    % Jingyi Wu, 2025
    if nargin < 1 || isempty(fontSize)
        fontSize = 13;
    end
    if nargin < 2 || isempty(lineWidth)
        lineWidth = 1.2;
    end
    set(gca,'FontWeight','bold','FontSize',fontSize,'LineWidth',lineWidth,'FontName','Arial');
end
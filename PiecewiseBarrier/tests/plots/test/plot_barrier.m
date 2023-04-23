% Plot barrier function

clc; clear; close all;

plot_barrier_2D()
find_minimum()

% Save plots
% print('barrier.pdf', '-dpdf', '-r300');
% print('barrier_1.png', '-dpng', '-r300');


function plot_barrier_2D()

    x1 = linspace(-1, 1, 100);

    % Bounds on polynomial kernel
    f = 1.0000000175772528*x1.^2 + 4.756832229341722e-13*x1  + ...
        9.043882221153276e-10;

    x_11 = linspace(-1, -0.6, 100);
    x_12 = linspace(-0.6, -0.2, 100);
    x_13 = linspace(-0.2, 0.2, 100);
    x_14 = linspace(0.2, 0.6, 100);
    x_15 = linspace(0.6, 1.0, 100);

    f1 = -0.6257646550079469*x_11 - 0.12888967897763098;
    f2 = -0.28029019767977764*x_12 + 0.09871002732834824;
    f3 = 1.8775313491588013e-12*x_13 + 4.999558666025719e-7;
    f4 = 0.2802901976852299*x_14 + 0.09871002741731587;
    f5 = 0.6257646550090096*x_15 - 0.12888967894350842;

    eta = 0.04;

    plot(x1, f, 'LineWidth', 3)
    hold on
    plot(x_11, f1, 'LineWidth', 3, 'Color', 'black')
    plot(x_12, f2, 'LineWidth', 3, 'Color', 'black')
    plot(x_13, f3, 'LineWidth', 3, 'Color', 'black')
    plot(x_14, f4, 'LineWidth', 3, 'Color', 'black')
    plot(x_15, f5, 'LineWidth', 3, 'Color', 'black')

    xline(-0.2,'m--','LineWidth',2);
    xline(+0.2,'m--','LineWidth',2);
    xline(-1.0,'r--','LineWidth',2);
    xline(+1.0,'r--','LineWidth',2);
    yline(eta,'m--','LineWidth',2)
    text(-.05,0.5,'$X_0$', 'Color' , 'm', 'Interpreter','latex',...
        'FontSize',16)
    text(+1.05,0.6,'$X_u$','Interpreter','latex', ...
            'Color', 'red', 'FontSize',16)
    text(-1.20,0.6,'$X_u$','Interpreter','latex', ...
            'Color', 'red', 'FontSize',16)
    text(-0.90,0.1,['$\eta = $', num2str(eta)],'Interpreter','latex',...
        'Color', 'magenta', 'FontSize',16)
    xlabel('x');
    ylabel('B(x)');
    grid on
    xlim([-1.25, 1.25])
    ylim([-.1, 2.5])
    labels = {'\color{blue} SOS B(x)', '\color{black} Piecewise B(x)'};
    legend(labels, 'Location', 'NorthWest', 'FontSize', 8, ...
    'TextColor', 'black');

end

function find_minimum()

    % Barrier function
    f = @(x1) 1.0000007651953218*x1.^2 - 1.7197334168473105e-11*x1 - ...
        9.088427152842553e-10;

    % Find the minimum of the function in the interval [-10,10]
    [x_min, f_min] = fminbnd(f,-10,40);
    
    % Display the result
    fprintf(['The minimum value of the function is f(x)=%f,' ...
        ' which occurs at x=%f.\n'],f_min,x_min);

end



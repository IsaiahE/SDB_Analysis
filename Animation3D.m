% 3d Animation
% Isaiah Ertel 
% Animates Dumbbell Path in Configuration Space
path = 'C:\Users\isaia\OneDrive - purdue.edu\Spinning Dumbbell Analysis\ER_1-20\IR-1_IT-1.5_IP-0_IV-0_IOT-0_IOP-20_D-10_K-2_ER-2_FR-0_M1-2_M2-2\';
data_name = 'AnimationData.csv';
Data = readtable(append(path,data_name), 'NumHeaderLines', 1);
DataArray = table2array(Data);
time = DataArray(:, 1);
radius = DataArray(:, 2);
theta = DataArray(:, 3);
phi = DataArray(:, 4);

speed_up_rate = 4;
time_length = round(length(time)/speed_up_rate);

x1 = zeros([time_length 1]);
y1 = zeros([time_length 1]);
z1 = zeros([time_length 1]);

for i=1:time_length
    if rem(i, speed_up_rate) == 0
        x1(i/speed_up_rate) = 2 * radius(i) * sin(theta(i)) * sin(phi(i));
        y1(i/speed_up_rate) = 2 * radius(i) * sin(theta(i)) * cos(phi(i));
        z1(i/speed_up_rate) = 2 * radius(i) * cos(theta(i));
    end
end

x2 = zeros([time_length 1]);
y2 = zeros([time_length 1]);
z2 = zeros([time_length 1]);

for i=1:time_length
    if rem(i, speed_up_rate) == 0
        x2(i/speed_up_rate) = -2 * radius(i) * sin(theta(i)) * sin(phi(i));
        y2(i/speed_up_rate) = -2 * radius(i) * sin(theta(i)) * cos(phi(i));
        z2(i/speed_up_rate) = -2 * radius(i) * cos(theta(i));
    end
end

maxy = max(y1);
miny = min(y1);
maxx = max(x1);
minx = min(x1);
maxz = max(z1);
minz = min(z1);

curve1 = animatedline('LineWidth', 1);
curve2 = animatedline('LineWidth', 1);
xlabel('x');
ylabel('y');
zlabel('z');
title('Dumbbell');
set(gca, 'XLim', [minx, maxx], 'YLim', [miny, maxy], 'Zlim', [minz maxz]);
view(43, 24);
hold on;
for i=1:time_length
    if isvalid(curve1)
        addpoints(curve1, x1(i), y1(i), z1(i));
        addpoints(curve2, x2(i), y2(i), z2(i));
    else
        break
    end
    head1 = scatter3(x1(i), y1(i), z1(i), 'filled', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');
    head2 = scatter3(x2(i), y2(i), z2(i), 'filled', 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g');
    drawnow
    % pause(0.00001);
    delete(head1);
    delete(head2);
end

plot3(x1, y1, z1, x2, y2, z2)
xlabel('x');
ylabel('y');
zlabel('z');
title('Dumbbell');
set(gca, 'XLim', [-130, 130], 'YLim', [-15, 15], 'Zlim', [-.7 .7]);
savefig(append(path,'MatLabAnimation.fig'));

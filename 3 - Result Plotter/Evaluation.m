clc;
clear all;
dataset = load('reduced_data.mat');
data = dataset.dataset;
 
pf1 = load('1 - 3610.mat');
pf2 = load('3610 - 4100.mat');

h = animatedline;
for k = 1: length(data(:,1))
    truth = data(k, :);
    t = strsplit(truth(2),':');
    
    time = strcat(t(1), t(2), t(3));
    addpoints(h, str2double(time),str2double(truth(3)));
    
    %%
    %drawnow
end

p = [pf1.particles_history; pf2.particles_history];

figure;
f = animatedline;
for k = 1: length(data(:,1))
%     if (k >= 40)
%         [best_particle, index] = max(str2double(pf2.particles_history((1 + (k - 1)*1000): (1 + (k - 1)*1000) + 1000, 3)));
%     else
%         [best_particle, index] = max(str2double(pf1.particles_history((1 + (k - 1)*1000): (1 + (k - 1)*1000) + 1000, 3)));
%     end

    

    t = strsplit(pf1.particles_history(index),':');
    time = strcat(t(1), t(2), t(3));
    addpoints(f, str2double(time),str2double(pf1.particles_history(index)), 'r');

    drawnow
end


% 
% for i = 1: length(data(:,1))
%     
%     for j = 1: 4100
%         if (j <= 3610)
%             truth = data(i,1);
% 
%             p_time = strsplit(p1(j, 1),':');
%             t_time = strsplit(truth(1),':');
% 
%             h_diff = abs(str2double(t_time(1)) - str2double(p_time(1)));
%             m_diff = abs(str2double(t_time(2)) - str2double(p_time(2)));
%             
%             
%         else
%         end
%         
%     end
% end
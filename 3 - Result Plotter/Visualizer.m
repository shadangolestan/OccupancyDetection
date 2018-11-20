clear all;
dataset = load('test_data.mat');
dataset = dataset.test_data;

temp = [];
for i = 1: length(dataset) %seperating date from time:
    %str = strsplit(dataset(i, 1), ' ');
     if (mod(i, 5) == 0)
        temp = [temp; strcat(dataset(i, 1), " ", dataset(i, 2))];
     end
end

dates = char(temp);

%dates = datetime(temp,'InputFormat','dd/MM/yyyy');

hist = load('hist.mat');
%hist2 = load('parthist(5818 - 7579).mat');
hist = hist.particles_history;
%hist2 = hist2.particles_history;

%hist = [hist;hist2];
partition = 1;
partnums = 1000;
best_elements = 200;
states = [];
thedates = [];
for i = 1: length(dataset(:,1))
    if (mod(i, 5) == 0)
        states = [states; str2double(dataset(i,3))];
        %thedates = [thedates; dates(i, :)];
    end
end

est_state = [];
error = [];
for i = 1: length(hist(:,1))/partnums
    temp = [];
    temp = ((i-1)*partnums + 1: i*partnums)';
    temp = [temp, hist((i-1)*partnums + 1: i*partnums,3)];
    temp = sortrows(temp, 2, 'descend');
    %bests = temp((1:length(temp)));
    %bests = temp((1:1000));
    %bests = temp((1:100));
    %bests = temp((1:50));
    bests = temp((1:5));
    %bests = temp((1:5));
    
    error = [error; mean(abs(states(i) - hist((bests), 2)))];
    
    est_state = [est_state; mean(hist((bests), 2))];
end

MSE = mse(est_state, states(1:length(est_state)));

hold on;
title(strcat('Estimating Number of Occupants Using Particle Filter | MSE = ',num2str(MSE)));
xlabel('Time');
ylabel('Number of Occupants');
%est_state = est_state(1:500);
dn=datenum(dates,'yyyy-mm-dd HH:MM');
plot(dn(1:length(est_state)/partition), states(1:length(est_state)/partition), '--');
plot(dn(1:length(est_state)/partition), smooth(est_state(1:length(est_state)/partition)));


%plot(dn,data(:,2));
set(gca,'XTick',dn(1):1:dn(length(est_state)/partition));

datetick('x','mm/dd', 'keepticks');


%plot(smooth(smooth(error)));
%legend('Real Occupancies','Absolute Error')
legend('Real Occupancies','PF Estimated Occupancies')
hold off;
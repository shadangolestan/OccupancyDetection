dataset = load('test_data.mat');
dataset = dataset.dataset;
hist = load('Hist.mat');
long = load('i.mat');
long = long.i;
%hist2 = load('parthist(5818 - 7579).mat');
hist = hist.particles_history;
%hist2 = hist2.particles_history;

%hist = [hist;hist2];

partnums = 1000;
best_elements = 50;
states = [];
for i = 1: length(dataset(:,1))
    if (mod(i, 5) == 0)
        states = [states; dataset(i,3)];
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
    bests = temp((1:300));
    %bests = temp((1:100));
    %bests = temp((1:50));
    %bests = temp((1:10));
    bests = temp((1:5));
    
    error = [error; mean(abs(states(i) - str2double(hist((bests), 2))))];
    
    
    est_state = [est_state; mean(hist((bests), 2))];
end

MSE = mse(est_state, states(1:length(est_state)));

hold on;
title(strcat('Absolute Error of Particle Filter Algorithm in Each State | MSE = ',num2str(MSE)));
xlabel('Time');
ylabel('Number of Occupants');
plot(states(1:length(est_state)));
plot(smooth(est_state));
%plot(smooth(smooth(error)));
legend('Real Occupancies','Absolute Error')
%legend('Real Occupancies','PF Estimated Occupancies')
hold off;
clc;
clear all;
%% initialization
dataset = load('test_data.mat');
dataset = dataset.test_data(1:length(dataset.test_data(:,1)), :);
% reduced_data = [];
% for i = 1: length(dataset.data(:,1))
%     if (mod(i, 5) == 0)
%         reduced_data = [reduced_data; dataset.data(i,:)];
%     end
% end
% 
% dataset = reduced_data;

sensor_model = load('SensorModel.mat');
motion_model = load('Motion_Model.mat');

motion_model = motion_model.Px;

motion_model(:, 1) = erase(motion_model(:, 1),':');
%motion_model(:, 1) = erase(motion_model(:, 1),'-');

dataset(:, 2) = erase(dataset(:, 2),':');
dataset(:, 1) = erase(dataset(:, 1),'-');

motion_model = str2double(motion_model);
dataset = str2double(dataset);

% Co2 = load('Co2.mat');
% Dmpr = load('Dmpr.mat');
% motion_model = load('Px.mat');

% Co2 = Co2.Co2;
% Dmpr = Dmpr.Dmpr;

PFnum = 0;
label_index = 3;

max_occ = length(unique(dataset(:, label_index + 3*PFnum)));
populate_num = 10000;

particles = populateParticles(max_occ, populate_num);
particles = str2double(particles);

% particles = load('1 - 499.mat');
% particles = particles.particles_history(4001: 5000, :);
particles_history = [];

%% Particle Filter Algo
for i = 1: length(dataset(:,1))
    %tic
    particles = movingParticles(particles, motion_model, dataset(i,:));
    %toc
    %tic
    particles = weightingParticles(particles, sensor_model, dataset(i,3), dataset(i, 2));
    %toc
    %tic
    particles = Resampling(particles);
    %toc
    
    if (isempty(particles))
        disp('NO!');
    end
    
    if (mod(i, 5) == 0)
        particles_history = [particles_history; particles];
    end
    
    if (mod(i, 5) == 0)
        disp(i);
    end
end
disp('end');
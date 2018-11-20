clc;
clear all;
train_percent = 0.6;


dataset = load('Dataset.mat');
% dataset(16561: 16730) = [];
dataset = dataset.data(1:length(dataset.data(:,1)), :); 

train_indices = 1: 29122;
train_indices = train_indices(1:length(train_indices)*train_percent);

train_data = dataset(train_indices, :);
dataset(train_indices, :) = [];
test_data = dataset;

sensorColumn = 4;
label_index = 3;
window_size = 30;
occnum = length(unique(train_data(:, label_index)));

%computeSensorModel( dataset, sensorColumn, label_index )
%ComputeMotionModel( dataset, matsize, truelabel )

temp_dataset = str2double(train_data);
[ M_VOC, V_VOC, M_NW, V_NW, M_BLE, V_BLE ] = computeSensorModel(temp_dataset, sensorColumn, label_index);
[Px] = ComputeMotionModel(train_data, temp_dataset, occnum, label_index, window_size);

disp('end');
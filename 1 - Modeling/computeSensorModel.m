function [ M_VOC, V_VOC, M_NW, V_NW, M_BLE, V_BLE ] = computeSensorModel( dataset, sensorColumn, label_index )
    dataset = sortrows(dataset, label_index);
    occnum = length(unique(dataset(:, label_index)));
    M_VOC = [];
    V_VOC = [];
    M_NW = [];
    V_NW = [];
    M_BLE = [];
    V_BLE = [];
    
    measured_occ = [];
    i = 1;
    while (i < length(dataset(:, 1)))
        if (length(measured_occ) == occnum)
            break;
        end
        if (isempty(find(measured_occ == dataset(i, label_index))))
            j = i;
            while (dataset(i, label_index) == dataset(j + 1, label_index))
                j = j + 1;
                if (j == length(dataset(:,1)) - 1)
                    j = j + 1;
                    break;
                end
            end

            %if (i ~= j)
            if (1)
                M_VOC  = [M_VOC, mean(dataset(floor(2*(j + i)/3) : j, sensorColumn))];
                V_VOC  = [V_VOC, var(dataset(floor(2*(j + i)/3) : j, sensorColumn))];
               
                M_NW = [M_NW, mean(dataset(floor((j + i)) : j, sensorColumn + 1))];
                V_NW = [V_NW, var(dataset(floor((j + i)) : j, sensorColumn + 1))];

                M_BLE = [M_BLE, mean(dataset(floor((j + i)) : j, sensorColumn + 2))];
                V_BLE = [V_BLE, var(dataset(floor((j + i)) : j, sensorColumn + 2))];

                measured_occ = [measured_occ, str2double(dataset(i, label_index))];
                disp(dataset(i, label_index));
            end
            
            i = j + 1;
        else
            i = i + 1;
        end
    end
end


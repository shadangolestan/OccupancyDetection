function [ particles ] = weightingParticles(particles, sensor_model, truth, true_time)
    i = 1;
    weight_index = length(particles(1, :));
    while (i < length(particles(:, 1)))
        w_VOC = 0.6;
        w_NW = 0.9;
        w_BLE = 1;
        w_time = 4;
        
        p_time = particles(i,1);
        t_time = true_time;
        
        timediff = 31 - (abs(floor(p_time/10000) - floor(t_time/10000)) + abs(floor(mod(p_time, 10000)/100) - floor(mod(t_time, 10000)/100)/100)/10);
        
        if (timediff < 0)
            disp('error');
        end
        
%         s = num2str(str2double(p_time).','%06d');
%         h = s(1:2);
%         m = s(3:4);
%         h_diff = abs(str2double(t_time(1)) - str2double(h));
%         m_diff = abs(str2double(t_time(2)) - str2double(m));
%         
%         timediff = 31;
%         if (h_diff ~= 0)
%             timediff = timediff - h_diff;
%         end
%         if (m_diff ~= 0)
%             timediff = timediff - m_diff/10;
%         end
        
        if (sensor_model.V_VOC(truth + 1) == 0)
            sensor_model.V_VOC(truth + 1) = 0.001;
        end

        if (sensor_model.V_NW(truth + 1) == 0)
            sensor_model.V_NW(truth + 1) = 0.001;
        end
        
        p_co2 = pdf('Normal', particles(i, 2), sensor_model.M_VOC(truth + 1), sensor_model.V_VOC(truth + 1));
        p_nw = pdf('Normal', particles(i, 2), sensor_model.M_NW(truth + 1), sensor_model.V_NW(truth + 1));
        p_ble = pdf('Normal', particles(i, 2), sensor_model.M_BLE(truth + 1), sensor_model.V_BLE(truth + 1));
        
        weight = w_VOC*p_co2 + w_NW*p_nw + w_time*timediff + w_BLE*p_ble;
        particles(i, weight_index) = weight;
        
        i = i + 1;
    end
end


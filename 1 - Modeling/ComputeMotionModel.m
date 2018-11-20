function [ Px ] = ComputeMotionModel( dataset, temp_dataset, matsize, truelabel, window_size )
    Px = [];
    i = 0;
    while (i < length(dataset(:, 1)))
         probs = zeros(1, matsize + 1);
        
        h = i + window_size;
        if (h >= length(dataset(:, 1)))
            h = length(dataset(:, 1));
        %else
            %h = i + window_size;
        end
        
        for j = i + 2: h
            %probs(str2double(dataset(j, truelabel)) + 1) = probs(str2double(dataset(j, truelabel)) + 1) + 1;
            
             if(temp_dataset(j, 3) == 0)
                 probs(1) = probs(1) + 1;
             elseif(temp_dataset(j, 3) == 1)
                 probs(2) = probs(2) + 1;
             elseif(temp_dataset(j, 3) == 2)
                 probs(3) = probs(3) + 1;
             elseif(temp_dataset(j, 3) == 3)
                 probs(4) = probs(4) + 1;
             elseif(temp_dataset(j, 3) == 4)
                 probs(5) = probs(5) + 1;
             elseif(temp_dataset(j, 3) == 5)
                 probs(6) = probs(6) + 1;
             elseif(temp_dataset(j, 3) == 6)
                 probs(7) = probs(7) + 1;
             elseif(temp_dataset(j, 3) == 7)
                 probs(8) = probs(8) + 1;
             end
        end
        probs = probs ./ (h - i - 1);
        Px = [Px; [dataset(i + 1, 1), dataset(i + 1, 2), dataset(i + 1, 3),  max(str2double(dataset((i + 1):h, 7))), dataset(i + 1, 8), probs]];
        %Px = [Px; [dataset(i + 1, 1), dataset(i + 1, truelabel), probs]];

        i = i + window_size;
        %i = i + 1;
    end
end


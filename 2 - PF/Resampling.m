function [ new_gen_particles ] = Resampling( particles )
    weight_index = length(particles(1, :));
    
    weights = particles(:, weight_index);
    new_gen = [];
    for i = 1: length(particles(:,1))
        candidate = RouletteWheelSelection(weights);
        new_gen = [new_gen; particles(candidate, :)];
%         if (str2double(particles(candidate, 3)) == 0)
%             particles(i, 3) = randi(str2double(particles(candidate, 3)) + 1, str2double(particles(candidate, 3)) + 2) - 1;
%         else
%             particles(i, 3) = randi(str2double(particles(candidate, 3)), str2double(particles(candidate, 3)) + 2) - 1;
%         end
        %particles(i, 3) = str2double(particles(candidate, 3));
    end
    particles = new_gen;
    randoms = randperm(length(particles(:,1)), length(particles(:,1))*0.1);
    particles(randoms, :) = [];
    for i = 1: length(randoms)
        hrs = randi([0 24], 1, 1);
        min = randi([0 60], 1, 1);
        sec = randi([0 60], 1, 1);
        
        time = min + 100*hrs;
        
%         S2 = strcat(num2str(hrs.','%02d'), num2str(min.','%02d'), num2str(sec.','%02d'));
%         S3 = {S2};
        
        particles = [particles; [time, randi([0 7], 1, 1)], 0];
    end
    
    new_gen_particles = particles;
end


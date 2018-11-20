function [ particles ] = populateParticles( max_occ, populate_num )
    particles = [];
    
    for i = 1: populate_num
        
        hrs = randi([0 24], 1, 1);
        min = randi([0 60], 1, 1);
        %sec = randi([0 60], 1, 1);
        
        S2 = strcat(num2str(hrs.','%02d'), num2str(min.','%02d'));
        S3 = {S2};
        
        particles = [particles; [string(S3), randi([0 max_occ], 1, 1)], 0];
    end
end


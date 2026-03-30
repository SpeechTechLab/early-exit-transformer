function plot_filter_evolution(results, fs, filter_field, title_prefix)
    % Plot how filter changes across frames
    
    num_freq_bins = 256;
    num_frames = length(results);
    
    H_matrix = zeros(num_freq_bins, num_frames);
    
    for i = 1:num_frames
        if isempty(results{i})
            continue;
        end
        
        try
            [H, ~] = freqz(1, results{i}.(filter_field), num_freq_bins, fs);
            H_matrix(:, i) = 20*log10(abs(H) + eps);
        catch
            H_matrix(:, i) = nan;
        end
    end
    
    f = linspace(0, fs/2, num_freq_bins);
    
    imagesc(1:num_frames, f, H_matrix);
    axis xy;
    colorbar;
    xlabel('Frame Number');
    ylabel('Frequency (Hz)');
    title(sprintf('%s Filter Evolution', title_prefix));
    ylim([0, 4000]);  % Focus on speech frequencies
    colormap('jet');
    caxis([-40, 40]);  % For older MATLAB versions (instead of clim)
end
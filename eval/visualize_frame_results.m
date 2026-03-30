function visualize_frame_results(x, fs, results, frame_indices, analysis_mode)
    % Visualize frame-by-frame analysis results
    
    figure('Position', [100, 100, 1400, 900]);
    
    % 1. Original signal with frame markers
    subplot(4, 2, [1, 2]);
    t = (0:length(x)-1) / fs;
    plot(t, x, 'Color', [0.5, 0.5, 0.5]);
    hold on;
    
    % Mark frame positions
    valid_results = ~cellfun(@isempty, results);
    valid_indices = frame_indices(valid_results);
    y_lim = [min(x), max(x)];
    
    for i = 1:min(20, length(valid_indices))  % Show first 20 frames
        x_pos = valid_indices(i)/fs;
        plot([x_pos, x_pos], y_lim, 'r--', 'LineWidth', 0.5);
    end
    
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(sprintf('Original Signal with Frame Markers (%s mode)', analysis_mode));
    grid on;
    
    % 2. Example frame - original and glottal flow
    subplot(4, 2, 3);
    frame_to_show = round(length(results) / 2);  % Middle frame
    if ~isempty(results{frame_to_show})
        frame_start = frame_indices(frame_to_show);
        frame_end = frame_start + results{frame_to_show}.frame_length - 1;
        t_frame = (frame_start:frame_end) / fs;
        plot(t_frame, x(frame_start:frame_end));
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(sprintf('Frame %d - Original', frame_to_show));
        grid on;
    end
    
    subplot(4, 2, 4);
    if ~isempty(results{frame_to_show})
        t_sg = (0:length(results{frame_to_show}.sg)-1) / fs;
        plot(t_sg, results{frame_to_show}.sg);
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(sprintf('Frame %d - Glottal Flow', frame_to_show));
        grid on;
    end
    
    % 3. Concatenated glottal flow (OLA - Overlap-Add)
    subplot(4, 2, [5, 6]);
    glottal_concat = reconstruct_signal(results, frame_indices, length(x), 'sg', fs);
    glottal_concat = detrend(glottal_concat, 'linear');
    t_concat = (0:length(glottal_concat)-1) / fs;
    plot(t_concat, glottal_concat);
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('Reconstructed Glottal Flow (All Frames)');
    grid on;
    
    % 4. Vocal tract filters across frames (spectrogram-like)
    subplot(4, 2, 7);
    plot_filter_evolution(results, fs, 'Hvt', 'Vocal Tract');
    
    % 5. Glottal filters across frames
    subplot(4, 2, 8);
    plot_filter_evolution(results, fs, 'Hg', 'Glottal');
    
end
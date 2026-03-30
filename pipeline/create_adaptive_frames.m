function [frames, frame_indices] = create_adaptive_frames(x, fs, pitch_info, frame_shift)
    % Create frames based on pitch periods
    % Each frame contains 2-3 pitch periods
    
    num_periods = 2;  % Number of pitch periods per frame
    
    % Interpolate pitch to sample level
    frame_shift_pitch = pitch_info.frame_shift;
    time_pitch = (0:length(pitch_info.f0)-1) * frame_shift_pitch / fs;
    time_signal = (0:length(x)-1) / fs;
    
    f0_interp = interp1(time_pitch, pitch_info.f0, time_signal, 'linear', 'extrap');
    f0_interp(f0_interp < 50) = 100;  % Fallback for unvoiced
    
    periods_interp = fs ./ f0_interp;
    
    % Create frames
    current_pos = 1;
    frames = {};
    frame_indices = [];
    frame_idx = 1;
    
    while current_pos + round(mean(periods_interp) * num_periods) < length(x)
        % Get local pitch period
        local_period = round(mean(periods_interp(current_pos:min(current_pos+100, length(periods_interp)))));
        frame_length = local_period * num_periods;
        
        end_pos = min(current_pos + frame_length - 1, length(x));
        
        frames{frame_idx} = x(current_pos:end_pos);
        frame_indices(frame_idx) = current_pos;
        
        % Shift by fixed amount
        current_pos = current_pos + frame_shift;
        frame_idx = frame_idx + 1;
    end
    
    frames = frames';
    frame_indices = frame_indices';
end
function [frames, frame_indices] = create_fixed_frames(x, frame_length, frame_shift)
    % Create overlapping frames with fixed length
    
    num_frames = floor((length(x) - frame_length) / frame_shift) + 1;
    frames = cell(num_frames, 1);
    frame_indices = zeros(num_frames, 1);
    
    for i = 1:num_frames
        start_idx = (i-1) * frame_shift + 1;
        end_idx = start_idx + frame_length - 1;
        
        if end_idx > length(x)
            break;
        end
        
        frames{i} = x(start_idx:end_idx);
        frame_indices(i) = start_idx;
    end
    
    % Remove empty cells
    frames = frames(~cellfun('isempty', frames));
    frame_indices = frame_indices(1:length(frames));
end
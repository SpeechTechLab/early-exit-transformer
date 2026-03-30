function signal_out = reconstruct_signal(results, frame_indices, total_length, field_name, fs)
    % Overlap-add reconstruction
    % Works for both QCP (sg) and IAIF (g)
    % If field_name not provided, auto-detects from results structure
    
    % Auto-detect field if not provided
    if nargin < 4 || isempty(field_name)
        field_name = [];
        % Auto-detect field name from first valid result
        for i = 1:length(results)
            if ~isempty(results{i})
                if isfield(results{i}, 'sg')
                    field_name = 'sg';  % QCP
                    break;
                elseif isfield(results{i}, 'g')
                    field_name = 'g';   % IAIF
                    break;
                end
            end
        end
        if isempty(field_name)
            error('Could not determine field name (sg or g)');
        end
    end

    if nargin < 6
        do_lowpass = false;
    end

    signal_out = zeros(total_length, 1);
    window_sum = zeros(total_length, 1);

    for i = 1:length(results)
        if isempty(results{i})
            continue;
        end

        frame_start = frame_indices(i);
        sg = results{i}.(field_name);
        sg = sg(:);  % column
        sg_length = length(sg);
        frame_end = frame_start + sg_length - 1;

        % Bounds check
        if frame_start < 1 || frame_start > total_length
            continue;
        end

        % Trim frame if it exceeds the total length
        if frame_end > total_length
            frame_end = total_length;
            actual_length = frame_end - frame_start + 1;
            sg = sg(1:actual_length);
            sg_length = actual_length;
        end

        % Synthesis window (use same type as in analysis if you want)
        win = hann(sg_length, 'periodic');
        windowed_frame = sg .* win;

        target_range = frame_start:frame_end;
        if length(windowed_frame) ~= length(target_range)
            warning('Frame %d: dimension mismatch, skipping', i);
            continue;
        end

        signal_out(target_range) = signal_out(target_range) + windowed_frame;
        window_sum(target_range) = window_sum(target_range) + win;
    end

    % Avoid division by (almost) zero at regions with little/no overlap
    window_sum(window_sum < 1e-3) = 1;
    signal_out = signal_out ./ window_sum;

    % Optional low-pass to smooth high-frequency artifacts
    if nargin >= 5 && ~isempty(fs)
        fc = 1500; % 1.5 kHz
        [b, a] = butter(2, fc / (fs / 2), 'low');
        signal_out = filtfilt(b, a, signal_out);
    end
end
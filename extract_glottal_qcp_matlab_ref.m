% extract_glottal_qcp_matlab_ref.m
% MATLAB reference extraction for QCP on Glottal_signals_db
addpath("core", "eval", "framework", "pipeline");

data_dir = 'Glottal_signals_db';
out_csv = 'features_Glottal_signals_db_QCP_matlab.csv';

wav_files = dir(fullfile(data_dir, '**', '*.wav'));
if isempty(wav_files)
    error('No wav files found in %s', data_dir);
end

results = table();

metrics_list = {'NAQ', 'QOQ', 'HRF', 'H1H2', 'G_RMS', 'G_ZCR', 'G_CREST', 'DG_PEAK', 'RES_RMS', 'RES_LEN_RATIO'};

for w = 1:length(wav_files)
    file_path = fullfile(wav_files(w).folder, wav_files(w).name);
    [~, file_name, ~] = fileparts(wav_files(w).name);
    fprintf('File %d/%d: %s\n', w, length(wav_files), file_name);

    try
        [x, fs] = audioread(file_path);
        if size(x, 2) > 1
            x = mean(x, 2);
        end
        x = x(:);

        if skewness(x) < 0
            x = -x;
        end

        [b_hp, a_hp] = butter(2, 50/(fs/2), 'high');
        x = filtfilt(b_hp, a_hp, x);

        frame_length = round(0.050 * fs);
        frame_shift = round(0.010 * fs);
        [frames, ~] = create_fixed_frames(x, frame_length, frame_shift);
        num_frames = length(frames);
        if num_frames == 0
            continue;
        end

        pitch_info = estimate_pitch(x, fs);
        f0_valid = pitch_info.f0(pitch_info.f0 > 0);
        if isempty(f0_valid)
            global_f0 = 120;
        else
            global_f0 = median(f0_valid);
        end

        voiced_mask = pitch_info.f0 > 50;
        if length(voiced_mask) < num_frames
            voiced_mask = [voiced_mask; false(num_frames - length(voiced_mask), 1)];
        else
            voiced_mask = voiced_mask(1:num_frames);
        end

        options = struct();
        options.f0 = global_f0;
        options.causality = 0;
        options.remove_real_poles = 1;
        options.dq = 0.4;
        options.pq = 0.05;
        options.nramp = round(fs/8000*7);

        NAQ_all = nan(num_frames,1);
        QOQ_all = nan(num_frames,1);
        HRF_all = nan(num_frames,1);
        H1H2_all = nan(num_frames,1);
        G_RMS_all = nan(num_frames,1);
        G_ZCR_all = nan(num_frames,1);
        G_CREST_all = nan(num_frames,1);
        DG_PEAK_all = nan(num_frames,1);
        RES_RMS_all = nan(num_frames,1);
        RES_LEN_RATIO_all = nan(num_frames,1);

        for i = 1:num_frames
            frame_data = frames{i};
            try
                frame_signal = signal(frame_data, fs);
                [sg, ~, e_ar, ~] = qcp(frame_signal, options);
                g_flow = sg.s(:);

                if isempty(g_flow)
                    continue;
                end

                metrics = compute_glottal_metrics(g_flow, fs, global_f0);
                NAQ_all(i) = metrics.NAQ;
                QOQ_all(i) = metrics.QOQ;
                HRF_all(i) = metrics.HRF;
                H1H2_all(i) = metrics.H1H2;

                g_centered = g_flow - mean(g_flow);
                g_rms = sqrt(mean(g_centered.^2));
                G_RMS_all(i) = g_rms;
                G_ZCR_all(i) = sum(abs(diff(g_centered > 0))) / max(length(g_centered)-1, 1);
                G_CREST_all(i) = max(abs(g_centered)) / max(g_rms, eps);
                DG_PEAK_all(i) = max(abs(diff(g_centered)));

                if isa(e_ar, 'signal')
                    residual_frame = e_ar.s(:);
                else
                    residual_frame = e_ar;
                end

                if ~isempty(residual_frame)
                    residual_frame = residual_frame(:);
                    if numel(residual_frame) > 1
                        RES_RMS_all(i) = sqrt(mean(residual_frame.^2));
                        RES_LEN_RATIO_all(i) = numel(residual_frame) / max(numel(frame_data), 1);
                    else
                        RES_RMS_all(i) = abs(residual_frame(1));
                        RES_LEN_RATIO_all(i) = NaN;
                    end
                end
            catch
                continue;
            end
        end

        data_list = {NAQ_all, QOQ_all, HRF_all, H1H2_all, G_RMS_all, G_ZCR_all, G_CREST_all, DG_PEAK_all, RES_RMS_all, RES_LEN_RATIO_all};

        row = table();
        row.file_name = string(file_name);
        row.speaker = "unknown";
        row.label = "unknown";
        row.task = "Glottal_signals_db";

        for m = 1:length(metrics_list)
            m_name = metrics_list{m};
            vals = data_list{m};
            vals = vals(voiced_mask);
            vals = vals(~isnan(vals));

            if isempty(vals)
                row.(sprintf('%s_mean', m_name)) = NaN;
                row.(sprintf('%s_std', m_name)) = NaN;
                row.(sprintf('%s_skewness', m_name)) = NaN;
                row.(sprintf('%s_kurtosis', m_name)) = NaN;
                row.(sprintf('%s_q25', m_name)) = NaN;
                row.(sprintf('%s_q50', m_name)) = NaN;
                row.(sprintf('%s_q75', m_name)) = NaN;
            else
                row.(sprintf('%s_mean', m_name)) = mean(vals);
                row.(sprintf('%s_std', m_name)) = std(vals);
                row.(sprintf('%s_skewness', m_name)) = skewness(vals);
                row.(sprintf('%s_kurtosis', m_name)) = kurtosis(vals);
                row.(sprintf('%s_q25', m_name)) = quantile(vals, 0.25);
                row.(sprintf('%s_q50', m_name)) = quantile(vals, 0.50);
                row.(sprintf('%s_q75', m_name)) = quantile(vals, 0.75);
            end
        end

        results = [results; row];

        if mod(w, 25) == 0
            writetable(results, out_csv);
            fprintf('Checkpoint save: %s (%d rows)\n', out_csv, height(results));
        end

    catch ME
        fprintf('Error processing %s: %s\n', file_name, ME.message);
    end
end

writetable(results, out_csv);
fprintf('Saved MATLAB reference QCP features to %s (%d rows)\n', out_csv, height(results));

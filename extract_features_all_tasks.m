% extract_features_all_tasks.m
addpath("core","eval","framework","pipeline");

data_dir = 'data/PC-GITA_per_task_44100Hz';
tasks = dir(data_dir);
tasks = tasks([tasks.isdir]);
tasks = tasks(~ismember({tasks.name}, {'.', '..'}));

% Filter to only process the 'Vowels' task
tasks = tasks(strcmp({tasks.name}, 'Vowels'));

if isempty(tasks)
    error('Task "Vowels" not found in the data directory.');
end

algorithms = {'IAIF', 'QCP', 'TRLP'};

for t = 1:length(tasks)
    task_name = tasks(t).name;
    fprintf('Processing task: %s\n', task_name);
    
    task_dir = fullfile(data_dir, task_name);
    
    % Find all wav files recursively
    wav_files = dir(fullfile(task_dir, '**', '*.wav'));
    
    if isempty(wav_files)
        fprintf('No wav files found in %s\n', task_name);
        continue;
    end
    
    % Initialize results tables for each algorithm and load existing data
    results_tables = struct();
    processed_files = struct();
    
    for a = 1:length(algorithms)
        algo = algorithms{a};
        csv_filename = sprintf('features_%s_%s.csv', task_name, algo);
        
        if isfile(csv_filename)
            % Load existing table
            results_tables.(algo) = readtable(csv_filename);

            % Normalize metadata text columns to string to avoid VERTCAT type/width issues
            text_cols = {'file_name', 'speaker', 'label', 'task'};
            for tc = 1:length(text_cols)
                col_name = text_cols{tc};
                if ismember(col_name, results_tables.(algo).Properties.VariableNames)
                    results_tables.(algo).(col_name) = string(results_tables.(algo).(col_name));
                end
            end

            % Keep track of processed files for this algorithm
            if ismember('file_name', results_tables.(algo).Properties.VariableNames)
                processed_files.(algo) = string(results_tables.(algo).file_name);
            else
                processed_files.(algo) = strings(0,1);
            end
            fprintf('Loaded %d existing records for %s\n', height(results_tables.(algo)), algo);
        else
            results_tables.(algo) = table();
            processed_files.(algo) = strings(0,1);
        end
    end
    
    for w = 1:length(wav_files)
        file_path = fullfile(wav_files(w).folder, wav_files(w).name);
        [~, file_name, ~] = fileparts(wav_files(w).name);
        file_name_str = string(file_name);
        
        % Extract speaker ID and class from filename
        % e.g., AVPEPUDEAC0019 (Control) or AVPEPUDEAA0053 (Pathological)
        % Sometimes pathological files are just AVPEPUDEA0059 (missing the second A)
        match = regexp(file_name, 'AVPEPUDEA([AC]?\d{4})', 'tokens');
        if isempty(match)
            fprintf('Skipping file with unknown naming format: %s\n', file_name);
            continue;
        end
        speaker_id = match{1}{1};
        
        % If the speaker_id starts with a digit, it means the 'A' or 'C' was missing.
        % Based on the dataset, these are pathological (A).
        if isstrprop(speaker_id(1), 'digit')
            speaker_id = ['A', speaker_id];
        end
        
        label = speaker_id(1); % 'A' or 'C'
        
        % Check if this file has already been processed by ALL algorithms
        all_processed = true;
        for a = 1:length(algorithms)
            algo = algorithms{a};
            if ~ismember(file_name_str, processed_files.(algo))
                all_processed = false;
                break;
            end
        end
        
        if all_processed
            fprintf('  Skipping File %d/%d: %s (Already processed)\n', w, length(wav_files), file_name);
            continue;
        end
        
        fprintf('  File %d/%d: %s\n', w, length(wav_files), file_name);
        
        try
            [x, fs] = audioread(file_path);
            if size(x, 2) > 1
                x = mean(x, 2);
            end
            x = x(:);
            
            if skewness(x) < 0
                x = -x;
            end
            
            fc_hp = 50;
            [b_hp, a_hp] = butter(2, fc_hp/(fs/2), 'high');
            x = filtfilt(b_hp, a_hp, x);
            
            % Frame parameters
            frame_length_ms = 50;
            frame_shift_ms = 10;
            frame_shift = round(frame_shift_ms * fs / 1000);
            frame_length = round(frame_length_ms * fs / 1000);
            
            [frames, frame_indices] = create_fixed_frames(x, frame_length, frame_shift);
            num_frames = length(frames);
            
            % Estimate global f0 for the whole signal
            pitch_info = estimate_pitch(x, fs);
            global_f0 = median(pitch_info.f0(pitch_info.f0 > 0));
            if isnan(global_f0) || global_f0 == 0
                global_f0 = 120; % fallback
            end
            
            % Determine voiced frames mask (f0 > 50 Hz)
            voiced_mask = pitch_info.f0 > 50;
            
            % Extract MFCCs (13 coeffs + delta + delta-delta = 39 features)
            try
                [coeffs, delta, deltaDelta] = mfcc(x, fs, 'WindowLength', frame_length, 'OverlapLength', frame_length - frame_shift);
                mfcc_features = [coeffs, delta, deltaDelta];
                % Align mfcc frames with voiced_mask
                min_frames = min(length(voiced_mask), size(mfcc_features, 1));
                mfcc_voiced = mfcc_features(1:min_frames, :);
                mfcc_voiced = mfcc_voiced(voiced_mask(1:min_frames), :);
            catch
                mfcc_voiced = [];
            end
            
            % Initialize feature arrays for this file
            file_features = struct();
            for a = 1:length(algorithms)
                algo = algorithms{a};
                file_features.(algo) = struct();
                file_features.(algo).file_name = file_name_str;
                file_features.(algo).speaker = string(speaker_id);
                file_features.(algo).label = string(label);
                file_features.(algo).task = string(task_name);
                
                metrics_list = {'NAQ', 'QOQ', 'HRF', 'H1H2', 'G_RMS', 'G_ZCR', 'G_CREST', 'DG_PEAK', 'RES_RMS', 'RES_LEN_RATIO'};
                for m = 1:length(metrics_list)
                    m_name = metrics_list{m};
                    file_features.(algo).(sprintf('%s_mean', m_name)) = NaN;
                    file_features.(algo).(sprintf('%s_std', m_name)) = NaN;
                    file_features.(algo).(sprintf('%s_skewness', m_name)) = NaN;
                    file_features.(algo).(sprintf('%s_kurtosis', m_name)) = NaN;
                    file_features.(algo).(sprintf('%s_q25', m_name)) = NaN;
                    file_features.(algo).(sprintf('%s_q50', m_name)) = NaN;
                    file_features.(algo).(sprintf('%s_q75', m_name)) = NaN;
                end
                
                % Initialize MFCC stats
                for c = 1:39
                    file_features.(algo).(sprintf('mfcc_%d_mean', c)) = NaN;
                    file_features.(algo).(sprintf('mfcc_%d_std', c)) = NaN;
                end
            end
            
            for a = 1:length(algorithms)
                algo = algorithms{a};
                
                % Arrays to store metrics across frames
                NAQ_all = zeros(num_frames, 1);
                QOQ_all = zeros(num_frames, 1);
                HRF_all = zeros(num_frames, 1);
                H1H2_all = zeros(num_frames, 1);
                G_RMS_all = zeros(num_frames, 1);
                G_ZCR_all = zeros(num_frames, 1);
                G_CREST_all = zeros(num_frames, 1);
                DG_PEAK_all = zeros(num_frames, 1);
                RES_RMS_all = zeros(num_frames, 1);
                RES_LEN_RATIO_all = zeros(num_frames, 1);
                
                % Options for algorithms
                options = struct();
                options.f0 = global_f0;
                options.causality = 0;
                options.remove_real_poles = 1;
                
                if strcmp(algo, 'IAIF')
                    options.p = round(fs/1000 + 2);
                    options.g = 4;
                elseif strcmp(algo, 'QCP')
                    options.dq = 0.4;
                    options.pq = 0.05;
                    options.nramp = round(fs/8000*7);
                elseif strcmp(algo, 'TRLP')
                    p = round(fs/1000 + 2);
                    rho = 0.99;
                    a_prev = zeros(p, 1);
                end
                
                valid_frames = 0;
                
                for i = 1:num_frames
                    frame_data = frames{i};
                    g_flow = [];
                    residual_frame = [];
                    
                    try
                        if strcmp(algo, 'IAIF')
                            [g, ~, e_ar, ~] = iaif(frame_data, fs, options);
                            if isa(g, 'signal'), g = g.s; end
                            g_flow = g(:);
                            if isa(e_ar, 'signal')
                                residual_frame = e_ar.s;
                            else
                                residual_frame = e_ar;
                            end
                        elseif strcmp(algo, 'QCP')
                            frame_signal = signal(frame_data, fs);
                            [sg, ~, e_ar, ~] = qcp(frame_signal, options);
                            g_flow = sg.s;
                            if isa(e_ar, 'signal')
                                residual_frame = e_ar.s;
                            else
                                residual_frame = e_ar;
                            end
                        elseif strcmp(algo, 'TRLP')
                            a_coeff = mytrlp(frame_data, p, a_prev, 0.5);
                            a_prev = a_coeff;
                            residual = filter([1; -a_coeff], 1, frame_data);
                            g_flow = filter(1, [1 -rho], residual);
                            g_flow = g_flow(:);
                            residual_frame = residual;
                        end
                        
                        if ~isempty(g_flow)
                            metrics = compute_glottal_metrics(g_flow, fs, global_f0);
                            NAQ_all(i) = metrics.NAQ;
                            QOQ_all(i) = metrics.QOQ;
                            HRF_all(i) = metrics.HRF;
                            H1H2_all(i) = metrics.H1H2;

                            g_centered = g_flow(:) - mean(g_flow(:));
                            g_rms = sqrt(mean(g_centered.^2));
                            G_RMS_all(i) = g_rms;
                            G_ZCR_all(i) = sum(abs(diff(g_centered > 0))) / max(length(g_centered) - 1, 1);
                            G_CREST_all(i) = max(abs(g_centered)) / max(g_rms, eps);
                            DG_PEAK_all(i) = max(abs(diff(g_centered)));

                            if ~isempty(residual_frame)
                                residual_frame = residual_frame(:);
                                if numel(residual_frame) > 1
                                    RES_RMS_all(i) = sqrt(mean(residual_frame.^2));
                                    RES_LEN_RATIO_all(i) = numel(residual_frame) / max(numel(frame_data), 1);
                                else
                                    % Some methods return scalar error terms instead of residual vectors
                                    % Keep amplitude proxy and mark length ratio as NaN (not comparable in size)
                                    RES_RMS_all(i) = abs(residual_frame(1));
                                    RES_LEN_RATIO_all(i) = NaN;
                                end
                            else
                                RES_RMS_all(i) = NaN;
                                RES_LEN_RATIO_all(i) = NaN;
                            end

                            valid_frames = valid_frames + 1;
                        else
                            NAQ_all(i) = NaN; QOQ_all(i) = NaN; HRF_all(i) = NaN; H1H2_all(i) = NaN;
                            G_RMS_all(i) = NaN; G_ZCR_all(i) = NaN; G_CREST_all(i) = NaN; DG_PEAK_all(i) = NaN; RES_RMS_all(i) = NaN; RES_LEN_RATIO_all(i) = NaN;
                        end
                    catch
                        NAQ_all(i) = NaN; QOQ_all(i) = NaN; HRF_all(i) = NaN; H1H2_all(i) = NaN;
                        G_RMS_all(i) = NaN; G_ZCR_all(i) = NaN; G_CREST_all(i) = NaN; DG_PEAK_all(i) = NaN; RES_RMS_all(i) = NaN; RES_LEN_RATIO_all(i) = NaN;
                    end
                end
                
                % Keep only voiced frames and remove NaNs
                voiced_len = min(num_frames, length(voiced_mask));
                voiced_idx = false(num_frames, 1);
                voiced_idx(1:voiced_len) = voiced_mask(1:voiced_len);
                
                NAQ_all = NAQ_all(voiced_idx);
                QOQ_all = QOQ_all(voiced_idx);
                HRF_all = HRF_all(voiced_idx);
                H1H2_all = H1H2_all(voiced_idx);
                G_RMS_all = G_RMS_all(voiced_idx);
                G_ZCR_all = G_ZCR_all(voiced_idx);
                G_CREST_all = G_CREST_all(voiced_idx);
                DG_PEAK_all = DG_PEAK_all(voiced_idx);
                RES_RMS_all = RES_RMS_all(voiced_idx);
                RES_LEN_RATIO_all = RES_LEN_RATIO_all(voiced_idx);
                
                NAQ_all = NAQ_all(~isnan(NAQ_all));
                QOQ_all = QOQ_all(~isnan(QOQ_all));
                HRF_all = HRF_all(~isnan(HRF_all));
                H1H2_all = H1H2_all(~isnan(H1H2_all));
                G_RMS_all = G_RMS_all(~isnan(G_RMS_all));
                G_ZCR_all = G_ZCR_all(~isnan(G_ZCR_all));
                G_CREST_all = G_CREST_all(~isnan(G_CREST_all));
                DG_PEAK_all = DG_PEAK_all(~isnan(DG_PEAK_all));
                RES_RMS_all = RES_RMS_all(~isnan(RES_RMS_all));
                RES_LEN_RATIO_all = RES_LEN_RATIO_all(~isnan(RES_LEN_RATIO_all));
                
                % Compute statistics
                metrics_list = {'NAQ', 'QOQ', 'HRF', 'H1H2', 'G_RMS', 'G_ZCR', 'G_CREST', 'DG_PEAK', 'RES_RMS', 'RES_LEN_RATIO'};
                data_list = {NAQ_all, QOQ_all, HRF_all, H1H2_all, G_RMS_all, G_ZCR_all, G_CREST_all, DG_PEAK_all, RES_RMS_all, RES_LEN_RATIO_all};
                
                for m = 1:length(metrics_list)
                    m_name = metrics_list{m};
                    m_data = data_list{m};
                    
                    if isempty(m_data)
                        file_features.(algo).(sprintf('%s_mean', m_name)) = NaN;
                        file_features.(algo).(sprintf('%s_std', m_name)) = NaN;
                        file_features.(algo).(sprintf('%s_skewness', m_name)) = NaN;
                        file_features.(algo).(sprintf('%s_kurtosis', m_name)) = NaN;
                        file_features.(algo).(sprintf('%s_q25', m_name)) = NaN;
                        file_features.(algo).(sprintf('%s_q50', m_name)) = NaN;
                        file_features.(algo).(sprintf('%s_q75', m_name)) = NaN;
                    else
                        file_features.(algo).(sprintf('%s_mean', m_name)) = mean(m_data);
                        file_features.(algo).(sprintf('%s_std', m_name)) = std(m_data);
                        file_features.(algo).(sprintf('%s_skewness', m_name)) = skewness(m_data);
                        file_features.(algo).(sprintf('%s_kurtosis', m_name)) = kurtosis(m_data);
                        file_features.(algo).(sprintf('%s_q25', m_name)) = quantile(m_data, 0.25);
                        file_features.(algo).(sprintf('%s_q50', m_name)) = quantile(m_data, 0.50);
                        file_features.(algo).(sprintf('%s_q75', m_name)) = quantile(m_data, 0.75);
                    end
                end

                % MFCC statistics on voiced frames (added per algorithm file for fair comparison)
                if ~isempty(mfcc_voiced)
                    for c = 1:39
                        file_features.(algo).(sprintf('mfcc_%d_mean', c)) = mean(mfcc_voiced(:, c));
                        file_features.(algo).(sprintf('mfcc_%d_std', c)) = std(mfcc_voiced(:, c));
                    end
                end
            end
            
            for a = 1:length(algorithms)
                algo = algorithms{a};
                if ~ismember(file_name_str, processed_files.(algo))
                    results_tables.(algo) = [results_tables.(algo); struct2table(file_features.(algo))];
                    processed_files.(algo) = [processed_files.(algo); file_name_str];
                end
            end
            
            % Save periodically (e.g., every 10 files) to avoid losing data on crash
            if mod(w, 10) == 0
                for a = 1:length(algorithms)
                    algo = algorithms{a};
                    if ~isempty(results_tables.(algo))
                        writetable(results_tables.(algo), sprintf('features_%s_%s.csv', task_name, algo));
                    end
                end
            end
            
        catch ME
            fprintf('Error processing file %s: %s\n', file_name, ME.message);
        end
    end
    
    % Final save
    for a = 1:length(algorithms)
        algo = algorithms{a};
        if ~isempty(results_tables.(algo))
            writetable(results_tables.(algo), sprintf('features_%s_%s.csv', task_name, algo));
            fprintf('Saved features for task %s, algorithm %s\n', task_name, algo);
        end
    end
end

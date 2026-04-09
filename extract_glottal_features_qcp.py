import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft
from scipy.linalg import toeplitz
from scipy.signal import butter, filtfilt, lfilter
from scipy.stats import kurtosis, skew
import soundfile as sf


def create_fixed_frames(x, frame_length, frame_shift):
    num_frames = int(np.floor((len(x) - frame_length) / frame_shift) + 1)
    if num_frames <= 0:
        return [], np.array([], dtype=np.int64)

    frames = []
    frame_indices = []
    for i in range(num_frames):
        start_idx = i * frame_shift
        end_idx = start_idx + frame_length
        if end_idx > len(x):
            break
        frames.append(x[start_idx:end_idx])
        frame_indices.append(start_idx)
    return frames, np.asarray(frame_indices, dtype=np.int64)


def estimate_pitch(x, fs):
    frame_length = int(round(0.04 * fs))
    frame_shift = int(round(0.01 * fs))
    min_lag = int(round(fs / 500.0))
    max_lag = int(round(fs / 50.0))

    num_frames = int(np.floor((len(x) - frame_length) / frame_shift) + 1)
    if num_frames <= 0:
        return {"f0": np.array([], dtype=np.float64)}

    f0 = np.zeros(num_frames, dtype=np.float64)
    ham = np.hamming(frame_length)

    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_length
        if end > len(x):
            break

        frame = x[start:end] * ham
        acf = signal.correlate(frame, frame, mode="full")
        lags = np.arange(-len(frame) + 1, len(frame))
        center = len(acf) // 2

        lag_min = max(min_lag, 1)
        lag_max = min(max_lag, len(frame) - 1)
        if lag_min > lag_max:
            continue

        acf_pos = acf[center + lag_min:center + lag_max + 1]
        rel = np.max(np.abs(acf[center]))
        if rel > 0:
            acf_pos = acf_pos / rel

        if acf_pos.size == 0:
            continue

        peak_idx = int(np.argmax(acf_pos))
        period = lag_min + peak_idx
        if period > 0:
            f0[i] = fs / period

    if f0.size >= 5:
        f0 = signal.medfilt(f0, kernel_size=5)

    return {"f0": f0}


def lpc_coeffs(x, order):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    if len(x) <= order:
        return np.r_[1.0, np.zeros(order, dtype=np.float64)]

    r = signal.correlate(x, x, mode="full")
    mid = len(r) // 2
    r = r[mid:mid + order + 1]
    if np.allclose(r[0], 0.0):
        return np.r_[1.0, np.zeros(order, dtype=np.float64)]

    R = toeplitz(r[:-1])
    rhs = r[1:]
    try:
        a = np.linalg.solve(R, rhs)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(R, rhs, rcond=None)[0]

    return np.r_[1.0, -a]


def find_f0_yin(x, fs, f0_min=60.0, f0_max=2000.0, threshold=0.1):
    tau_min = int(np.floor(fs / f0_max))
    tau_max = int(np.ceil(fs / f0_min))

    W = len(x) - tau_max
    if W < 1:
        return 0.0

    d = np.zeros(tau_max, dtype=np.float64)
    base = x[:W]
    for tau in range(1, tau_max + 1):
        diff = base - x[tau:tau + W]
        d[tau - 1] = np.sum(diff * diff)

    csum = np.cumsum(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        dn = d / (csum / np.arange(1, tau_max + 1))
    dn[~np.isfinite(dn)] = 1.0

    start = max(tau_min - 1, 0)
    end = max(tau_max - 1, start + 1)
    dn_range = dn[start:end]
    if dn_range.size < 3:
        idx = int(np.argmin(dn_range)) + start
        return fs / max(idx + 1, 1)

    diff_dn = np.diff(dn)
    d1 = diff_dn[start - 1:end - 1] if start > 0 else diff_dn[:end - 1]
    d2 = diff_dn[start:end]
    local = dn[start:end]
    n = min(len(d1), len(d2), len(local))
    d1 = d1[:n]
    d2 = d2[:n]
    local = local[:n]

    minima = np.where((d1 <= 0) & (d2 > 0) & (local <= threshold))[0]
    if minima.size > 0:
        min_idx = int(minima[0] + start + 1)
    else:
        min_idx = int(np.argmin(dn_range) + start + 1)

    if 1 < min_idx < len(dn):
        y = dn[min_idx - 1:min_idx + 2]
        A = np.array([[0.5, -1.0, 0.5], [-0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
        a, b, _ = A @ y
        if abs(a) > 1e-12:
            x0 = -b / (2.0 * a)
            min_idx = min_idx + x0

    if min_idx <= 0:
        return 0.0
    return float(fs / min_idx)


def gci(x, meanf0, fs):
    x = np.asarray(x, dtype=np.float64).ravel()
    N = len(x)
    if N < 8:
        return np.array([1, N], dtype=np.int64), np.array([1, N], dtype=np.int64)

    if meanf0 <= 0:
        meanf0 = max(find_f0_yin(x, fs), 100.0)

    if 1.0 / meanf0 < 0.008:
        Tf0 = fs / meanf0
        Nw = int(round(1.4 * Tf0 / 2.0))
    else:
        Nw = int(round(0.006 * fs))
    Nw = max(Nw, 1)

    p = 18 if fs == 16000 else 10

    x2 = np.r_[np.zeros(Nw), x, np.zeros(Nw + 1)]
    w = np.blackman(2 * Nw + 1)
    y = np.zeros(N, dtype=np.float64)

    for n in range(N):
        seg = x2[n:n + 2 * Nw + 1]
        y[n] = np.sum(w * seg)
    y = y / (2 * Nw + 1)

    ydiff = np.gradient(y)

    goi_int = []
    gci_int = []
    for i in range(N - 1):
        if np.sign(ydiff[i] * ydiff[i + 1]) == -1:
            j = i
            while j < N - 1 and np.sign(y[j] * y[j + 1]) != -1:
                j += 1
            if np.sign(ydiff[i]) == 1:
                goi_int.append((i, j))
            else:
                gci_int.append((i, j))

    a = lpc_coeffs(x, p)
    x_res = lfilter(a, [1.0], x)

    gci_ins = []
    for s, e in gci_int:
        e = max(e, s + 1)
        seg = x_res[s:e + 1]
        idx = int(np.argmax(seg))
        gci_ins.append(s + idx)

    goi_ins = []
    for s, e in goi_int:
        e = max(e, s + 1)
        seg = x_res[s:e + 1]
        idx = int(np.argmax(seg))
        goi_ins.append(s + idx)

    if len(gci_ins) < 2:
        step = max(int(round(fs / max(meanf0, 80.0))), 1)
        gci_ins = list(range(0, N, step))
        if len(gci_ins) < 2:
            gci_ins = [0, N - 1]

    return np.asarray(gci_ins, dtype=np.int64), np.asarray(goi_ins, dtype=np.int64)


def makeW(x, p, DQ, PQ, d, Nramp, gci_ins):
    N = len(x)
    w = np.full(N + p, d, dtype=np.float64)

    if Nramp > 0:
        upramp = np.linspace(d, 1.0, 2 + Nramp)[1:-1]
        downramp = upramp[::-1]
    else:
        upramp = np.array([])
        downramp = np.array([])

    if DQ + PQ > 1:
        DQ = 1 - PQ

    if len(gci_ins) < 2:
        return w

    for i in range(len(gci_ins) - 1):
        T = int(gci_ins[i + 1] - gci_ins[i])
        if T <= 0:
            continue
        T1 = int(round(DQ * T))
        T2 = int(round(PQ * T))
        while T1 + T2 > T and T1 > 0:
            T1 -= 1

        start = int(gci_ins[i] + T2)
        end = int(start + T1)
        start = max(start, 0)
        end = min(end, N + p)
        if end > start:
            w[start:end] = 1.0

        if Nramp > 0 and end - start > Nramp:
            ur_end = min(start + Nramp, N + p)
            dr_start = max(end - Nramp, 0)
            w[start:ur_end] = upramp[: max(ur_end - start, 0)]
            w[dr_start:end] = downramp[-max(end - dr_start, 0):]

    return w


def wlp(x, w, p):
    N = len(x)
    Sn = np.zeros((p, N + p), dtype=np.float64)

    for n in range(N + p):
        for i in range(1, p + 1):
            idx = n - i
            if 0 <= idx < N:
                Sn[i - 1, n] = x[idx]

    R = np.zeros((p, p), dtype=np.float64)
    r = np.zeros((p,), dtype=np.float64)
    x2 = np.r_[x, np.zeros(p)]

    for i in range(N + p):
        si = Sn[:, i]
        R = R + w[i] * np.outer(si, si)
        r = r + w[i] * si * x2[i]

    try:
        a = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(R, r, rcond=None)[0]

    a = np.r_[1.0, -a]
    e_ar = float(np.sum(lfilter(a, [1.0], x) ** 2))
    return a, e_ar


def integrate_signal(x, rho=0.99, causality="noncausal"):
    x = np.asarray(x, dtype=np.float64).ravel()
    if causality.lower() == "causal":
        y = lfilter([1.0], [1.0, -rho], x)
    else:
        y = np.flip(lfilter([1.0], [1.0, -rho], np.flip(x)))
    return y


def qcp(frame, fs, options):
    x = np.asarray(frame, dtype=np.float64).ravel()
    f0 = float(options.get("f0", 0.0))
    if f0 <= 0:
        f0 = max(find_f0_yin(x, fs), 100.0)

    p = int(round(fs / 1000.0) + 2)
    rho = float(options.get("rho", 0.99))
    DQ = float(options.get("dq", 0.4))
    PQ = float(options.get("pq", 0.05))
    Nramp = int(options.get("nramp", round(fs / 8000.0 * 7)))
    causality = "causal" if int(options.get("causality", 0)) == 1 else "noncausal"
    remove_real_poles = int(options.get("remove_real_poles", 0))

    gci_ins, _ = gci(x, f0, fs)
    w = makeW(x, p, DQ, PQ, 1e-5, Nramp, gci_ins)

    s2 = lfilter([1.0, -1.0], [1.0], x)
    sw = np.hamming(len(s2)) * s2
    Hvt, e_ar = wlp(sw, w, p)

    if remove_real_poles:
        roots = np.roots(Hvt)
        roots = np.asarray([r for r in roots if (np.real(r) < 0) or (abs(np.imag(r)) > 1e-15)])
        if roots.size > 0:
            Hvt = np.poly(roots).real

    dg_temp = lfilter(Hvt, [1.0], x)
    if causality == "causal":
        dg_temp = -dg_temp

    sg = integrate_signal(dg_temp, rho=rho, causality=causality)
    return sg, Hvt, e_ar


def find_peak_near(spectrum, freqs, target_freq, bandwidth):
    mask = (freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(freqs - target_freq)))
        return float(spectrum[idx]), idx
    idxs = np.where(mask)[0]
    local = int(np.argmax(spectrum[idxs]))
    idx = int(idxs[local])
    return float(spectrum[idx]), idx


def compute_glottal_metrics(g, fs, f0):
    g = np.asarray(g, dtype=np.float64).ravel()
    if f0 <= 0:
        return {"NAQ": np.nan, "QOQ": np.nan, "HRF": np.nan, "H1H2": np.nan}

    T0_sec = 1.0 / f0
    T0_samples = int(round(fs / f0))
    if len(g) < max(T0_samples, 2):
        return {"NAQ": np.nan, "QOQ": np.nan, "HRF": np.nan, "H1H2": np.nan}

    dg = np.r_[np.diff(g), 0.0] * fs
    g_detr = signal.detrend(g)
    A_ac = float(np.max(g_detr) - np.min(g_detr))

    min_peak_height = float(np.max(np.abs(dg)) * 0.2)
    pks, _ = signal.find_peaks(-dg, height=min_peak_height, distance=max(int(round(T0_samples * 0.8)), 1))
    if pks.size == 0:
        E_e = float(np.max(np.abs(dg)))
    else:
        E_e = float(np.mean((-dg)[pks]))

    naq = A_ac / (E_e * T0_sec) if E_e > 0 else np.nan

    threshold = float(np.min(g_detr) + 0.5 * A_ac)
    above = g_detr > threshold
    d_above = np.diff(np.r_[0, above.astype(np.int32), 0])
    starts = np.where(d_above == 1)[0]
    ends = np.where(d_above == -1)[0]
    if starts.size and ends.size:
        lengths = ends - starts
        valid = lengths[(lengths > 0.1 * T0_samples) & (lengths < 1.1 * T0_samples)]
        qoq = float(np.mean(valid) / T0_samples) if valid.size else np.nan
    else:
        qoq = np.nan

    nfft = 1 << int(np.ceil(np.log2(max(4 * len(g), 2))))
    w = np.hamming(len(g))
    G = fft(g * w, n=nfft)
    mag = np.abs(G)
    freqs = np.arange(len(mag)) * fs / len(mag)

    h1, _ = find_peak_near(mag, freqs, f0, 0.1 * f0)
    h2, _ = find_peak_near(mag, freqs, 2.0 * f0, 0.1 * f0)

    harmonics_sum = 0.0
    k = 2
    while k * f0 < fs / 2.0:
        hk, _ = find_peak_near(mag, freqs, k * f0, 0.1 * f0)
        harmonics_sum += hk
        k += 1

    if h1 > 0 and h2 > 0:
        hrf = 20.0 * np.log10(max(harmonics_sum, 1e-12) / h1)
        h1h2 = 20.0 * np.log10(h1) - 20.0 * np.log10(h2)
    else:
        hrf = np.nan
        h1h2 = np.nan

    return {"NAQ": naq, "QOQ": qoq, "HRF": hrf, "H1H2": h1h2}


def safe_stats(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    return (
        float(np.mean(arr)),
        float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        float(skew(arr, bias=False)) if arr.size > 2 else 0.0,
        float(kurtosis(arr, fisher=False, bias=False)) if arr.size > 3 else 0.0,
        float(np.quantile(arr, 0.25)),
        float(np.quantile(arr, 0.50)),
        float(np.quantile(arr, 0.75)),
    )


def compute_matlab_like_mfcc(x, fs, frame_length, frame_shift, voiced_mask):
    """Approximate MATLAB Audio Toolbox mfcc() settings using librosa.

    Key choices:
    - 40 mel bands (MATLAB default filter bank size)
    - Hamming analysis window
    - natural-log mel energies before the DCT (instead of librosa's default dB scaling)
    - 13 static MFCCs + delta + delta-delta with width=9
    """
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=x,
            sr=fs,
            n_fft=frame_length,
            hop_length=frame_shift,
            win_length=frame_length,
            window=signal.windows.hamming(frame_length, sym=False),
            center=False,
            power=2.0,
            n_mels=40,
            fmin=133.3333,
            fmax=min(6864.0, fs / 2.0),
            htk=False,
            norm="slaney",
        )
        log_mel_spec = np.log(np.maximum(mel_spec, np.finfo(np.float64).eps))
        coeffs = librosa.feature.mfcc(
            S=log_mel_spec,
            n_mfcc=13,
            dct_type=2,
            norm="ortho",
            lifter=0,
        )
        delta = librosa.feature.delta(coeffs, width=9, order=1, mode="nearest")
        delta_delta = librosa.feature.delta(coeffs, width=9, order=2, mode="nearest")
        mfcc_all = np.concatenate([coeffs, delta, delta_delta], axis=0).T  # (n_frames, 39)
        n_align = min(len(voiced_mask), mfcc_all.shape[0])
        return mfcc_all[:n_align][voiced_mask[:n_align]]
    except Exception:
        return np.empty((0, 39))


def extract_file_qcp(file_path):
    x, fs = sf.read(file_path)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    x = x.astype(np.float64).ravel()
    if len(x) < 32:
        return None

    if skew(x, bias=False) < 0:
        x = -x

    b_hp, a_hp = butter(2, 50.0 / (fs / 2.0), btype="high")
    x = filtfilt(b_hp, a_hp, x)

    frame_length = int(round(0.050 * fs))
    frame_shift = int(round(0.010 * fs))

    frames, _ = create_fixed_frames(x, frame_length, frame_shift)
    num_frames = len(frames)
    if num_frames == 0:
        return None

    pitch_info = estimate_pitch(x, fs)
    f0v = pitch_info["f0"]
    valid_f0 = f0v[f0v > 0]
    global_f0 = float(np.median(valid_f0)) if valid_f0.size else 120.0

    voiced_mask = f0v > 50
    if voiced_mask.size < num_frames:
        vm = np.zeros(num_frames, dtype=bool)
        vm[:voiced_mask.size] = voiced_mask
        voiced_mask = vm
    else:
        voiced_mask = voiced_mask[:num_frames]

    options = {
        "f0": global_f0,
        "causality": 0,
        "remove_real_poles": 1,
        "dq": 0.4,
        "pq": 0.05,
        "nramp": int(round(fs / 8000.0 * 7)),
    }

    NAQ_all = np.full(num_frames, np.nan)
    QOQ_all = np.full(num_frames, np.nan)
    HRF_all = np.full(num_frames, np.nan)
    H1H2_all = np.full(num_frames, np.nan)
    G_RMS_all = np.full(num_frames, np.nan)
    G_ZCR_all = np.full(num_frames, np.nan)
    G_CREST_all = np.full(num_frames, np.nan)
    DG_PEAK_all = np.full(num_frames, np.nan)
    RES_RMS_all = np.full(num_frames, np.nan)
    RES_LEN_RATIO_all = np.full(num_frames, np.nan)

    for i, frame in enumerate(frames):
        try:
            g_flow, _, residual = qcp(frame, fs, options)
            if g_flow is None or len(g_flow) == 0:
                continue

            metrics = compute_glottal_metrics(g_flow, fs, global_f0)
            NAQ_all[i] = metrics["NAQ"]
            QOQ_all[i] = metrics["QOQ"]
            HRF_all[i] = metrics["HRF"]
            H1H2_all[i] = metrics["H1H2"]

            g_centered = g_flow - np.mean(g_flow)
            g_rms = np.sqrt(np.mean(g_centered ** 2))
            G_RMS_all[i] = g_rms
            G_ZCR_all[i] = np.sum(np.abs(np.diff(g_centered > 0))) / max(len(g_centered) - 1, 1)
            G_CREST_all[i] = np.max(np.abs(g_centered)) / max(g_rms, np.finfo(float).eps)
            DG_PEAK_all[i] = np.max(np.abs(np.diff(g_centered))) if len(g_centered) > 1 else np.nan

            if np.isscalar(residual):
                RES_RMS_all[i] = abs(float(residual))
                RES_LEN_RATIO_all[i] = np.nan
            else:
                residual = np.asarray(residual, dtype=np.float64).ravel()
                if residual.size > 1:
                    RES_RMS_all[i] = np.sqrt(np.mean(residual ** 2))
                    RES_LEN_RATIO_all[i] = residual.size / max(frame.size, 1)
                elif residual.size == 1:
                    RES_RMS_all[i] = abs(float(residual[0]))
                    RES_LEN_RATIO_all[i] = np.nan
        except Exception:
            continue

    def voiced_clean(a):
        return a[voiced_mask]

    feature_data = {
        "NAQ": voiced_clean(NAQ_all),
        "QOQ": voiced_clean(QOQ_all),
        "HRF": voiced_clean(HRF_all),
        "H1H2": voiced_clean(H1H2_all),
        "G_RMS": voiced_clean(G_RMS_all),
        "G_ZCR": voiced_clean(G_ZCR_all),
        "G_CREST": voiced_clean(G_CREST_all),
        "DG_PEAK": voiced_clean(DG_PEAK_all),
        "RES_RMS": voiced_clean(RES_RMS_all),
        "RES_LEN_RATIO": voiced_clean(RES_LEN_RATIO_all),
    }

    row = {}
    for key, vals in feature_data.items():
        m, s, sk, ku, q25, q50, q75 = safe_stats(vals)
        row[f"{key}_mean"] = m
        row[f"{key}_std"] = s
        row[f"{key}_skewness"] = sk
        row[f"{key}_kurtosis"] = ku
        row[f"{key}_q25"] = q25
        row[f"{key}_q50"] = q50
        row[f"{key}_q75"] = q75

    # MFCC extraction disabled.
    # To restore it later, uncomment the block below.
    # mfcc_voiced = compute_matlab_like_mfcc(x, fs, frame_length, frame_shift, voiced_mask)
    #
    # for c in range(1, 40):
    #     vals = mfcc_voiced[:, c - 1] if mfcc_voiced.shape[0] > 0 else np.array([])
    #     finite_vals = vals[np.isfinite(vals)]
    #     row[f"mfcc_{c}_mean"] = float(np.mean(finite_vals)) if finite_vals.size > 0 else np.nan
    #     row[f"mfcc_{c}_std"] = float(np.std(finite_vals)) if finite_vals.size > 0 else np.nan

    file_stem = Path(file_path).stem
    row["file_name"] = file_stem
    row["speaker"] = "unknown"
    row["label"] = "unknown"
    row["task"] = "Glottal_signals_db"
    return row


def run(input_dir, output_csv, max_files=0, save_every=50, resume=False):
    wavs = sorted(Path(input_dir).rglob("*.wav"))
    if max_files and max_files > 0:
        wavs = wavs[:max_files]

    rows = []
    done = set()
    if resume and Path(output_csv).exists():
        try:
            prev = pd.read_csv(output_csv)
            if "file_name" in prev.columns:
                done = set(prev["file_name"].astype(str))
                rows = prev.to_dict(orient="records")
                print(f"Resuming from {output_csv} with {len(done)} completed files")
        except Exception as exc:
            print(f"Warning: could not resume from {output_csv}: {exc}")

    for idx, wav in enumerate(wavs, start=1):
        if wav.stem in done:
            continue
        print(f"[{idx}/{len(wavs)}] {wav}")
        row = extract_file_qcp(str(wav))
        if row is not None:
            rows.append(row)
            done.add(wav.stem)

        if save_every > 0 and len(rows) % save_every == 0:
            df_ckpt = pd.DataFrame(rows)
            meta_cols_ckpt = ["file_name", "speaker", "label", "task"]
            other_cols_ckpt = [c for c in df_ckpt.columns if c not in meta_cols_ckpt]
            df_ckpt = df_ckpt[meta_cols_ckpt + other_cols_ckpt]
            df_ckpt.to_csv(output_csv, index=False)
            print(f"Checkpoint saved: {len(df_ckpt)} rows -> {output_csv}")

    if not rows:
        raise RuntimeError("No features extracted; check input files")

    df = pd.DataFrame(rows)
    meta_cols = ["file_name", "speaker", "label", "task"]
    other_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + other_cols]
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Extract QCP glottal features from wav files (Python port)")
    parser.add_argument("--input_dir", default="Glottal_signals_db", help="Directory with wav files")
    parser.add_argument("--output_csv", default="features_Glottal_signals_db_QCP_python.csv", help="Output CSV")
    parser.add_argument("--max_files", type=int, default=0, help="Optional limit for quick tests")
    parser.add_argument("--save_every", type=int, default=50, help="Checkpoint frequency (rows)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output_csv")
    args = parser.parse_args()

    run(
        args.input_dir,
        args.output_csv,
        max_files=args.max_files,
        save_every=args.save_every,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

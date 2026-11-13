# VitalsLens-AI

Sprint 1 provides live MediaPipe face and pose landmarking, spatial facial ROI
RGB extraction, an illumination reference ROI, and a strict face-presence gate.
Sprint 2 adds face-gated signal buffering, uniform resampling, per-region POS
projection, zero-phase bandpass filtering, FFT spectra, and SNR diagnostics.
Sprint 3 adds SNR-weighted regional BVP fusion, an SQI/artifact gate, and
experimental HR, HRV, respiratory-rate, and calibrated optical SpO₂ proxy
estimates. These estimates are research-only and are not for clinical use.

## Run locally

```sh
npm install
npm run dev
```

Allow camera access when prompted. MediaPipe WASM and float16 task models are
loaded from the official CDN and Google-hosted model URLs at runtime. DSP
plots warm up after 90 valid frames; vital estimates require at least 150
valid frames and an acceptable SQI. The Sprint 2 camera viewport remains
uncropped and uses full-frame pixels for ROI and background-reference math.

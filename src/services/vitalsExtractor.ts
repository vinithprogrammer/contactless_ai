import {
  bandpassZeroPhase,
  computePowerSpectrumInBand,
  computeSpectrum,
} from './dspMath.ts'
import type { SignalFusionResult } from './signalFusion'
import type { SignalSample } from '../utils/ringBuffer'

export type AutonomicStress =
  | 'Optimal / Parasympathetic Recovery'
  | 'Balanced / Normal Tone'
  | 'Elevated Sympathetic Stress'

export interface OpticalCalibration {
  a: number
  b: number
}

export interface ExtractedVitals {
  heartRateBpm: number | null
  spectralHeartRateBpm: number | null
  ibiHeartRateBpm: number | null
  peakIndices: number[]
  rmssdMs: number | null
  autonomicStress: AutonomicStress | null
  respiratoryRateBrpm: number | null
  spo2ProxyPercent: number | null
  rRatio: number | null
  ibiConsistencyPercent: number | null
  confidencePercent: number | null
}

export const MIN_VITAL_SAMPLE_COUNT = 150
export const DEFAULT_OPTICAL_CALIBRATION: OpticalCalibration = {
  a: 104,
  b: 17,
}

const SAMPLE_RATE = 30
const MIN_PEAK_DISTANCE = Math.ceil(SAMPLE_RATE / 3)
const IBI_OUTLIER_FRACTION = 0.22

function mean(values: readonly number[]): number {
  return values.length === 0
    ? 0
    : values.reduce((sum, value) => sum + value, 0) / values.length
}

function median(values: readonly number[]): number {
  if (values.length === 0) return 0
  const ordered = [...values].sort((a, b) => a - b)
  const middle = Math.floor(ordered.length / 2)
  return ordered.length % 2 === 0
    ? (ordered[middle - 1] + ordered[middle]) / 2
    : ordered[middle]
}

function standardDeviation(values: readonly number[]): number {
  const center = mean(values)
  return Math.sqrt(mean(values.map((value) => (value - center) ** 2)))
}

function rootMeanSquare(values: readonly number[]): number {
  return Math.sqrt(mean(values.map((value) => value * value)))
}

export function detectSystolicPeaks(signal: readonly number[]): number[] {
  if (signal.length < MIN_PEAK_DISTANCE * 2) return []
  const center = mean(signal)
  const adaptiveProminence = Math.max(standardDeviation(signal) * 0.35, 1e-8)
  const candidates: Array<{ index: number; value: number; prominence: number }> = []
  const localRadius = Math.max(2, Math.round(SAMPLE_RATE * 0.2))

  for (let index = 1; index < signal.length - 1; index += 1) {
    const value = signal[index]
    if (value <= signal[index - 1] || value < signal[index + 1]) continue
    if (value < center + adaptiveProminence * 0.15) continue
    const start = Math.max(0, index - localRadius)
    const end = Math.min(signal.length, index + localRadius + 1)
    let localMinimum = Number.POSITIVE_INFINITY
    for (let cursor = start; cursor < end; cursor += 1) {
      localMinimum = Math.min(localMinimum, signal[cursor])
    }
    const prominence = value - localMinimum
    if (prominence < adaptiveProminence) continue
    candidates.push({ index, value, prominence })
  }

  const accepted: Array<{ index: number; value: number; prominence: number }> = []
  for (const candidate of candidates) {
    const last = accepted.at(-1)
    if (!last || candidate.index - last.index >= MIN_PEAK_DISTANCE) {
      accepted.push(candidate)
    } else if (
      candidate.value > last.value ||
      (candidate.value === last.value && candidate.prominence > last.prominence)
    ) {
      accepted[accepted.length - 1] = candidate
    }
  }
  return accepted.map(({ index }) => index)
}

export function calculateIbiMetrics(
  peakIndices: readonly number[],
  sampleRate = SAMPLE_RATE,
): {
  intervalsMs: number[]
  validIntervalsMs: number[]
  ibiHeartRateBpm: number | null
  rmssdMs: number | null
  ibiConsistencyPercent: number | null
} {
  const intervalsMs = peakIndices.slice(1).map(
    (peak, index) => ((peak - peakIndices[index]) / sampleRate) * 1000,
  )
  if (intervalsMs.length === 0) {
    return {
      intervalsMs,
      validIntervalsMs: [],
      ibiHeartRateBpm: null,
      rmssdMs: null,
      ibiConsistencyPercent: null,
    }
  }
  const centerIbi = median(intervalsMs)
  const validIntervalsMs = intervalsMs.filter(
    (interval) =>
      Math.abs(interval - centerIbi) / centerIbi <= IBI_OUTLIER_FRACTION,
  )
  const ibiHeartRateBpm =
    validIntervalsMs.length > 0
      ? 60_000 / median(validIntervalsMs)
      : null
  const validIntervalIndices = intervalsMs.map(
    (interval) =>
      Math.abs(interval - centerIbi) / centerIbi <= IBI_OUTLIER_FRACTION,
  )
  let longestValidRun: number[] = []
  let currentValidRun: number[] = []
  intervalsMs.forEach((interval, index) => {
    if (validIntervalIndices[index]) {
      currentValidRun.push(interval)
      if (currentValidRun.length > longestValidRun.length) {
        longestValidRun = [...currentValidRun]
      }
    } else {
      currentValidRun = []
    }
  })
  const successiveDifferences = longestValidRun
    .slice(1)
    .map((interval, index) => interval - longestValidRun[index])
  const rmssdMs =
    longestValidRun.length >= 4
      ? Math.sqrt(mean(successiveDifferences.map((difference) => difference ** 2)))
      : null

  return {
    intervalsMs,
    validIntervalsMs,
    ibiHeartRateBpm,
    rmssdMs,
    ibiConsistencyPercent: Math.round(
      (100 * validIntervalsMs.length) / intervalsMs.length,
    ),
  }
}

function classifyStress(rmssdMs: number | null): AutonomicStress | null {
  if (rmssdMs === null) return null
  if (rmssdMs > 45) return 'Optimal / Parasympathetic Recovery'
  if (rmssdMs >= 25) return 'Balanced / Normal Tone'
  return 'Elevated Sympathetic Stress'
}

function calculateRespiratoryRate(
  samples: readonly SignalSample[],
  fusedBvp: readonly number[],
): number | null {
  const shoulder = samples.map((sample) => sample.shoulderY)
  const observed = shoulder
    .map((value, index) => (value === null ? null : { index, value }))
    .filter((entry): entry is { index: number; value: number } => entry !== null)
  if (observed.length < MIN_VITAL_SAMPLE_COUNT / 2 || fusedBvp.length < 60) {
    return null
  }

  const shoulderFilled = [...shoulder]
  for (let index = 0; index < shoulderFilled.length; index += 1) {
    if (shoulderFilled[index] !== null) continue
    let before = index - 1
    while (before >= 0 && shoulderFilled[before] === null) before -= 1
    let after = index + 1
    while (after < shoulderFilled.length && shoulderFilled[after] === null) {
      after += 1
    }
    if (before >= 0 && after < shoulderFilled.length) {
      const ratio = (index - before) / (after - before)
      shoulderFilled[index] =
        (shoulderFilled[before] as number) +
        ((shoulderFilled[after] as number) -
          (shoulderFilled[before] as number)) *
          ratio
    } else if (before >= 0) {
      shoulderFilled[index] = shoulderFilled[before]
    } else if (after < shoulderFilled.length) {
      shoulderFilled[index] = shoulderFilled[after]
    }
  }

  const shoulderSignal = shoulderFilled.map((value) => value ?? 0)
  const mechanical = bandpassZeroPhase(
    shoulderSignal,
    SAMPLE_RATE,
    0.1,
    0.5,
  )
  const envelopeRadius = Math.max(1, Math.round(SAMPLE_RATE * 0.25))
  const envelope = fusedBvp.map((_, index) => {
    let sumSquares = 0
    for (let offset = -envelopeRadius; offset <= envelopeRadius; offset += 1) {
      let sampleIndex = index + offset
      if (sampleIndex < 0) sampleIndex = -sampleIndex
      if (sampleIndex >= fusedBvp.length) {
        sampleIndex = 2 * fusedBvp.length - sampleIndex - 2
      }
      const value = fusedBvp[sampleIndex]
      sumSquares += value * value
    }
    return Math.sqrt(sumSquares / (2 * envelopeRadius + 1))
  })
  const rsa = bandpassZeroPhase(envelope, SAMPLE_RATE, 0.1, 0.5)
  const mechanicalSpectrum =
    standardDeviation(shoulderSignal) > 1e-8
      ? computePowerSpectrumInBand(mechanical, SAMPLE_RATE, 0.1, 0.5)
      : []
  const rsaSpectrum =
    mean(envelope) > 1e-8 && rootMeanSquare(rsa) / mean(envelope) >= 0.015
      ? computePowerSpectrumInBand(rsa, SAMPLE_RATE, 0.1, 0.5)
      : []
  if (mechanicalSpectrum.length === 0 && rsaSpectrum.length === 0) return null

  const maxMechanical = Math.max(...mechanicalSpectrum.map((point) => point.power), 1e-12)
  const maxRsa = Math.max(...rsaSpectrum.map((point) => point.power), 1e-12)
  const bins = mechanicalSpectrum.length > 0 ? mechanicalSpectrum : rsaSpectrum
  const combined = bins.map((point) => {
    const mechanicalPower = nearestPower(mechanicalSpectrum, point.frequencyHz)
    const rsaPower = nearestPower(rsaSpectrum, point.frequencyHz)
    return {
      frequencyHz: point.frequencyHz,
      power: mechanicalPower / maxMechanical + rsaPower / maxRsa,
    }
  })
  const peak = combined.reduce(
    (best, point) => (point.power > best.power ? point : best),
    { frequencyHz: 0, power: 0 },
  )
  return peak.power > 0 ? Math.round(peak.frequencyHz * 60) : null
}

function nearestPower(
  spectrum: readonly { frequencyHz: number; power: number }[],
  targetHz: number,
): number {
  if (spectrum.length === 0) return 0
  let nearest = spectrum[0]
  for (const point of spectrum) {
    if (
      Math.abs(point.frequencyHz - targetHz) <
      Math.abs(nearest.frequencyHz - targetHz)
    ) {
      nearest = point
    }
  }
  return nearest.power
}

function calculateOxygenProxy(
  samples: readonly SignalSample[],
  fusedBvp: readonly number[],
  calibration: OpticalCalibration,
): { spo2ProxyPercent: number | null; rRatio: number | null } {
  const red = samples.map(({ forehead }) => forehead.r)
  const blue = samples.map(({ forehead }) => forehead.b)
  const redDc = mean(red)
  const blueDc = mean(blue)
  if (redDc <= 0 || blueDc <= 0 || fusedBvp.length < 3) {
    return { spo2ProxyPercent: null, rRatio: null }
  }
  const redAcSignal = bandpassZeroPhase(red, SAMPLE_RATE)
  const blueAcSignal = bandpassZeroPhase(blue, SAMPLE_RATE)
  const redAc = Math.sqrt(mean(redAcSignal.map((value) => value ** 2)))
  const blueAc = Math.sqrt(mean(blueAcSignal.map((value) => value ** 2)))
  if (redAc <= 1e-8 || blueAc <= 1e-8) {
    return { spo2ProxyPercent: null, rRatio: null }
  }
  const rRatio = (redAc / redDc) / (blueAc / blueDc)
  if (!Number.isFinite(rRatio)) return { spo2ProxyPercent: null, rRatio: null }
  const calibrated = Math.round((calibration.a - calibration.b * rRatio) * 10) / 10
  return {
    spo2ProxyPercent: Math.max(88, Math.min(100, calibrated)),
    rRatio,
  }
}

export function extractVitals(
  samples: readonly SignalSample[],
  fusion: SignalFusionResult,
  previousSmoothedBpm: number | null,
  calibration: OpticalCalibration = DEFAULT_OPTICAL_CALIBRATION,
): ExtractedVitals {
  const unavailable: ExtractedVitals = {
    heartRateBpm: null,
    spectralHeartRateBpm: null,
    ibiHeartRateBpm: null,
    peakIndices: [],
    rmssdMs: null,
    autonomicStress: null,
    respiratoryRateBrpm: null,
    spo2ProxyPercent: null,
    rRatio: null,
    ibiConsistencyPercent: null,
    confidencePercent: null,
  }
  if (
    samples.length < MIN_VITAL_SAMPLE_COUNT ||
    fusion.fusedBvp.length < MIN_VITAL_SAMPLE_COUNT
  ) {
    return unavailable
  }

  const spectrum = computeSpectrum(fusion.fusedBvp, SAMPLE_RATE)
  const spectralHeartRateBpm =
    spectrum.peakFrequencyHz === null ? null : spectrum.peakFrequencyHz * 60
  const peakIndices = detectSystolicPeaks(fusion.fusedBvp)
  const ibi = calculateIbiMetrics(peakIndices, SAMPLE_RATE)
  const estimates = [spectralHeartRateBpm, ibi.ibiHeartRateBpm].filter(
    (value): value is number => value !== null && Number.isFinite(value),
  )
  const rawBpm = estimates.length > 0 ? mean(estimates) : null
  const heartRateBpm =
    rawBpm === null
      ? null
      : previousSmoothedBpm === null
        ? rawBpm
        : previousSmoothedBpm + 0.25 * (rawBpm - previousSmoothedBpm)
  const oxygen = calculateOxygenProxy(samples, fusion.fusedBvp, calibration)
  const concentration =
    spectrum.snrDb === null
      ? 0
      : Math.max(0, Math.min(1, (spectrum.snrDb + 5) / 30))
  const agreement =
    spectralHeartRateBpm !== null && ibi.ibiHeartRateBpm !== null
      ? Math.max(
          0,
          100 -
            (Math.abs(spectralHeartRateBpm - ibi.ibiHeartRateBpm) /
              Math.max(spectralHeartRateBpm, ibi.ibiHeartRateBpm)) *
              100,
        )
      : 0

  return {
    heartRateBpm,
    spectralHeartRateBpm,
    ibiHeartRateBpm: ibi.ibiHeartRateBpm,
    peakIndices,
    rmssdMs: ibi.rmssdMs,
    autonomicStress: classifyStress(ibi.rmssdMs),
    respiratoryRateBrpm: calculateRespiratoryRate(samples, fusion.fusedBvp),
    spo2ProxyPercent: oxygen.spo2ProxyPercent,
    rRatio: oxygen.rRatio,
    ibiConsistencyPercent: ibi.ibiConsistencyPercent,
    confidencePercent: Math.round(0.65 * concentration * 100 + 0.35 * agreement),
  }
}

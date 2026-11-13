import { computeSpectrum } from './dspMath.ts'

export const SQI_MINIMUM_FOR_VITALS = 35
export const HEAD_MOTION_THRESHOLD = 0.02

export type ArtifactState =
  | 'READY'
  | 'NO_FACE'
  | 'MOTION_ARTIFACT'
  | 'LOW_LIGHT'
  | 'LOW_SQI'

export interface SqiGateInput {
  isFaceValid: boolean
  fusedBvp: readonly number[]
  recentHeadVelocities: readonly (number | null)[]
  foreheadMeanRgb: { r: number; g: number; b: number } | null
}

export interface SqiGateResult {
  sqiScore: number
  state: ArtifactState
  message: string
  spectralConcentration: number
  headStability: number
  illuminationAdequacy: number
  facialLuminance: number | null
  recentMaxHeadVelocity: number | null
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value))
}

function computeSpectralConcentration(signal: readonly number[]): number {
  const spectrum = computeSpectrum(signal, 30)
  const peak = spectrum.peakFrequencyHz
  if (peak === null || spectrum.points.length === 0) return 0
  const totalPower = spectrum.points.reduce((sum, point) => sum + point.power, 0)
  const peakBandPower = spectrum.points.reduce(
    (sum, point) =>
      Math.abs(point.frequencyHz - peak) <= 0.15 ? sum + point.power : sum,
    0,
  )
  return totalPower > 0 ? clamp01(peakBandPower / totalPower) : 0
}

export function evaluateSignalQuality(input: SqiGateInput): SqiGateResult {
  const validVelocities = input.recentHeadVelocities.filter(
    (value): value is number => value !== null && Number.isFinite(value),
  )
  const recentMaxHeadVelocity =
    validVelocities.length > 0 ? Math.max(...validVelocities) : null
  const headStability =
    recentMaxHeadVelocity === null
      ? 0
      : clamp01(1 - recentMaxHeadVelocity / HEAD_MOTION_THRESHOLD)

  const foreheadMeanRgb = input.foreheadMeanRgb
  const facialLuminance = foreheadMeanRgb
    ? 0.299 * foreheadMeanRgb.r +
      0.587 * foreheadMeanRgb.g +
      0.114 * foreheadMeanRgb.b
    : null
  const illuminationAdequacy =
    facialLuminance === null
      ? 0
      : clamp01((facialLuminance - 30) / 70)
  const spectralConcentration = computeSpectralConcentration(input.fusedBvp)
  const sqiScore = Math.round(
    100 *
      (0.6 * spectralConcentration +
        0.25 * headStability +
        0.15 * illuminationAdequacy),
  )

  let state: ArtifactState = 'READY'
  let message = 'SIGNAL QUALITY ACCEPTABLE'
  if (!input.isFaceValid) {
    state = 'NO_FACE'
    message = 'NO FACE DETECTED — HOLD IN FRAME'
  } else if (
    recentMaxHeadVelocity !== null &&
    recentMaxHeadVelocity > HEAD_MOTION_THRESHOLD
  ) {
    state = 'MOTION_ARTIFACT'
    message = 'MOTION ARTIFACT DETECTED — HOLD STILL'
  } else if (facialLuminance === null || facialLuminance < 40) {
    state = 'LOW_LIGHT'
    message = 'INSUFFICIENT AMBIENT LIGHTING'
  } else if (sqiScore < SQI_MINIMUM_FOR_VITALS) {
    state = 'LOW_SQI'
    message = 'STABILIZING SIGNAL QUALITY...'
  }

  return {
    sqiScore,
    state,
    message,
    spectralConcentration,
    headStability,
    illuminationAdequacy,
    facialLuminance,
    recentMaxHeadVelocity,
  }
}

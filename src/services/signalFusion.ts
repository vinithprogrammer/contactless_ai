import type { FacialRegion } from './posProcessor'

export interface RegionalSnr {
  forehead: number | null
  leftCheek: number | null
  rightCheek: number | null
}

export interface FusionWeights {
  forehead: number
  leftCheek: number
  rightCheek: number
}

export interface SignalFusionResult {
  weights: FusionWeights
  percentages: FusionWeights
  fusedBvp: number[]
}

const REGION_ORDER: readonly FacialRegion[] = [
  'forehead',
  'leftCheek',
  'rightCheek',
]
const LOW_SNR_WEIGHT_FACTOR = 0.001

export function calculateFusionWeights(snr: RegionalSnr): FusionWeights {
  const linearPower = REGION_ORDER.map((region) => {
    const value = snr[region]
    if (value === null || !Number.isFinite(value)) return 0
    const boundedDb = Math.max(-100, Math.min(60, value))
    const power = 10 ** (boundedDb / 10)
    return boundedDb < -3 ? power * LOW_SNR_WEIGHT_FACTOR : power
  })
  const totalPower = linearPower.reduce((sum, power) => sum + power, 0)
  const normalized =
    totalPower > 0
      ? linearPower.map((power) => power / totalPower)
      : [1 / 3, 1 / 3, 1 / 3]
  return {
    forehead: normalized[0],
    leftCheek: normalized[1],
    rightCheek: normalized[2],
  }
}

export function fuseBvpSignals(
  signals: Record<FacialRegion, readonly number[]>,
  weights: FusionWeights,
): number[] {
  const length = Math.min(
    signals.forehead.length,
    signals.leftCheek.length,
    signals.rightCheek.length,
  )
  return Array.from({ length }, (_, index) =>
    weights.forehead * signals.forehead[index] +
    weights.leftCheek * signals.leftCheek[index] +
    weights.rightCheek * signals.rightCheek[index],
  )
}

export function fuseRegionalBvp(
  snr: RegionalSnr,
  signals: Record<FacialRegion, readonly number[]>,
): SignalFusionResult {
  const weights = calculateFusionWeights(snr)
  return {
    weights,
    percentages: {
      forehead: weights.forehead * 100,
      leftCheek: weights.leftCheek * 100,
      rightCheek: weights.rightCheek * 100,
    },
    fusedBvp: fuseBvpSignals(signals, weights),
  }
}

import {
  bandpassZeroPhase,
  computeSpectrum,
  type SpectrumResult,
} from './dspMath.ts'
import type { RGB } from '../utils/roiExtractor'
import type { SignalSample } from '../utils/ringBuffer'

export const FACIAL_REGIONS = ['forehead', 'leftCheek', 'rightCheek'] as const
export type FacialRegion = (typeof FACIAL_REGIONS)[number]

export interface ProcessedRegion {
  pos: number[]
  filtered: number[]
  spectrum: SpectrumResult
}

export interface PosProcessingResult {
  rawGreen: number[]
  regions: Record<FacialRegion, ProcessedRegion>
}

const POS_WINDOW_LENGTH = 48
const POS_WINDOW_STEP = POS_WINDOW_LENGTH / 2
const MIN_CHANNEL_MEAN = 1e-6

function getWindowStarts(length: number): number[] {
  if (length < POS_WINDOW_LENGTH) return []
  const starts: number[] = []
  for (let start = 0; start + POS_WINDOW_LENGTH <= length; start += POS_WINDOW_STEP) {
    starts.push(start)
  }
  const finalStart = length - POS_WINDOW_LENGTH
  if (starts[starts.length - 1] !== finalStart) starts.push(finalStart)
  return starts
}

function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function standardDeviation(values: readonly number[]): number {
  const center = mean(values)
  return Math.sqrt(
    values.reduce((sum, value) => sum + (value - center) ** 2, 0) /
      values.length,
  )
}

function regionRgb(sample: SignalSample, region: FacialRegion): RGB {
  return sample[region]
}

export function projectPos(
  samples: readonly SignalSample[],
  region: FacialRegion,
): number[] {
  const output = Array.from({ length: samples.length }, () => 0)
  const overlapCount = Array.from({ length: samples.length }, () => 0)

  for (const start of getWindowStarts(samples.length)) {
    const window = samples.slice(start, start + POS_WINDOW_LENGTH)
    const channels = {
      r: window.map((sample) => regionRgb(sample, region).r),
      g: window.map((sample) => regionRgb(sample, region).g),
      b: window.map((sample) => regionRgb(sample, region).b),
    }
    const channelMeans = {
      r: Math.max(mean(channels.r), MIN_CHANNEL_MEAN),
      g: Math.max(mean(channels.g), MIN_CHANNEL_MEAN),
      b: Math.max(mean(channels.b), MIN_CHANNEL_MEAN),
    }
    const backgroundLuminance = window.map(
      ({ background }) =>
        0.299 * background.r + 0.587 * background.g + 0.114 * background.b,
    )
    const backgroundMean = Math.max(
      mean(backgroundLuminance),
      MIN_CHANNEL_MEAN,
    )
    const normalizedBackgroundDerivative = backgroundLuminance.map(
      (luminance, index) =>
        index === 0
          ? 0
          : (luminance - backgroundLuminance[index - 1]) / backgroundMean,
    )
    const normalized = channels.r.map((_, index) => ({
      r:
        channels.r[index] / channelMeans.r -
        normalizedBackgroundDerivative[index],
      g:
        channels.g[index] / channelMeans.g -
        normalizedBackgroundDerivative[index],
      b:
        channels.b[index] / channelMeans.b -
        normalizedBackgroundDerivative[index],
    }))
    const s1 = normalized.map(({ g, b }) => g - b)
    const s2 = normalized.map(({ r, g, b }) => g + b - 2 * r)
    const alpha =
      standardDeviation(s1) /
      Math.max(standardDeviation(s2), MIN_CHANNEL_MEAN)
    const pulse = s1.map((value, index) => value + alpha * s2[index])
    const pulseMean = mean(pulse)

    pulse.forEach((value, index) => {
      const sampleIndex = start + index
      output[sampleIndex] += value - pulseMean
      overlapCount[sampleIndex] += 1
    })
  }

  return output.map((value, index) =>
    overlapCount[index] === 0 ? 0 : value / overlapCount[index],
  )
}

function normalizedGreen(
  samples: readonly SignalSample[],
  region: FacialRegion,
): number[] {
  const green = samples.map((sample) => regionRgb(sample, region).g)
  const channelMean = mean(green)
  if (channelMean <= MIN_CHANNEL_MEAN) {
    return Array.from({ length: green.length }, () => 0)
  }
  return green.map((value) => value / channelMean - 1)
}

export function processPosRegions(
  samples: readonly SignalSample[],
  sampleRate = 30,
): PosProcessingResult {
  const processRegion = (region: FacialRegion): ProcessedRegion => {
    const pos = projectPos(samples, region)
    const filtered = bandpassZeroPhase(pos, sampleRate)
    return { pos, filtered, spectrum: computeSpectrum(filtered, sampleRate) }
  }
  const regions: Record<FacialRegion, ProcessedRegion> = {
    forehead: processRegion('forehead'),
    leftCheek: processRegion('leftCheek'),
    rightCheek: processRegion('rightCheek'),
  }

  return {
    rawGreen: normalizedGreen(samples, 'forehead'),
    regions,
  }
}

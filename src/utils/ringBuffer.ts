import type { RGB } from './roiExtractor'

export interface SignalSample {
  timestamp: number
  forehead: RGB
  leftCheek: RGB
  rightCheek: RGB
  background: RGB
  shoulderY: number | null
  headVelocity: number | null
}

export const SIGNAL_BUFFER_CAPACITY = 256
export const SIGNAL_BUFFER_WARMUP = 90
export const SIGNAL_GAP_RESET_MS = 1000

const RGB_CHANNELS = ['r', 'g', 'b'] as const

function isValidRgb(rgb: RGB): boolean {
  return RGB_CHANNELS.every((channel) => Number.isFinite(rgb[channel]))
}

function interpolateRgb(start: RGB, end: RGB, ratio: number): RGB {
  return {
    r: start.r + (end.r - start.r) * ratio,
    g: start.g + (end.g - start.g) * ratio,
    b: start.b + (end.b - start.b) * ratio,
  }
}

function interpolateNullable(
  start: number | null,
  end: number | null,
  ratio: number,
): number | null {
  return start === null || end === null
    ? null
    : start + (end - start) * ratio
}

export class SignalBuffer {
  private readonly storage: Array<SignalSample | undefined>
  private startIndex = 0
  private sampleCount = 0
  private lastValidSampleTime: number | null = null
  readonly capacity: number

  constructor(capacity = SIGNAL_BUFFER_CAPACITY) {
    if (!Number.isInteger(capacity) || capacity < 2) {
      throw new RangeError('SignalBuffer capacity must be an integer of at least 2.')
    }
    this.capacity = capacity
    this.storage = Array.from({ length: capacity }, () => undefined)
  }

  get length(): number {
    return this.sampleCount
  }

  get ready(): boolean {
    return this.sampleCount >= SIGNAL_BUFFER_WARMUP
  }

  push(
    sample: SignalSample | null,
    isFaceValid: boolean,
    now = performance.now(),
  ): boolean {
    if (
      this.lastValidSampleTime !== null &&
      now - this.lastValidSampleTime > SIGNAL_GAP_RESET_MS
    ) {
      this.clear()
    }

    if (
      !isFaceValid ||
      !sample ||
      !Number.isFinite(sample.timestamp) ||
      !isValidRgb(sample.forehead) ||
      !isValidRgb(sample.leftCheek) ||
      !isValidRgb(sample.rightCheek) ||
      !isValidRgb(sample.background) ||
      (sample.shoulderY !== null && !Number.isFinite(sample.shoulderY)) ||
      (sample.headVelocity !== null && !Number.isFinite(sample.headVelocity))
    ) {
      return false
    }

    const previous = this.sampleCount > 0
      ? this.storage[(this.startIndex + this.sampleCount - 1) % this.capacity]
      : undefined
    if (previous && sample.timestamp <= previous.timestamp) return false

    const insertionIndex =
      this.sampleCount < this.capacity
        ? (this.startIndex + this.sampleCount) % this.capacity
        : this.startIndex
    this.storage[insertionIndex] = { ...sample }
    if (this.sampleCount < this.capacity) this.sampleCount += 1
    else this.startIndex = (this.startIndex + 1) % this.capacity
    this.lastValidSampleTime = now
    return true
  }

  clear(): void {
    this.storage.fill(undefined)
    this.startIndex = 0
    this.sampleCount = 0
    this.lastValidSampleTime = null
  }

  getSamples(): SignalSample[] {
    return Array.from({ length: this.sampleCount }, (_, index) => {
      const sample = this.storage[(this.startIndex + index) % this.capacity]
      if (!sample) throw new Error('SignalBuffer contains an unexpected empty slot.')
      return sample
    })
  }
}

export function resampleUniform(
  samples: readonly SignalSample[],
  targetFps = 30,
  length = SIGNAL_BUFFER_CAPACITY,
): SignalSample[] {
  if (!Number.isFinite(targetFps) || targetFps <= 0) {
    throw new RangeError('Resampling frame rate must be greater than zero.')
  }
  if (!Number.isInteger(length) || length < 1) {
    throw new RangeError('Resampling length must be a positive integer.')
  }
  if (samples.length === 0) return []

  for (let index = 1; index < samples.length; index += 1) {
    if (samples[index].timestamp <= samples[index - 1].timestamp) {
      throw new RangeError('Resampling samples must have increasing timestamps.')
    }
  }
  if (samples.length === 1) return [{ ...samples[0] }]

  const intervalMs = 1000 / targetFps
  const firstTime = samples[0].timestamp
  const lastTime = samples[samples.length - 1].timestamp
  const intervalCount = (lastTime - firstTime) / intervalMs
  const roundingTolerance =
    Number.EPSILON * Math.max(1, intervalCount) * 8
  const coveredIntervals = Math.floor(intervalCount + roundingTolerance)
  const outputLength = Math.min(length, coveredIntervals + 1)
  const startTime = lastTime - (outputLength - 1) * intervalMs
  const result: SignalSample[] = []
  let rightIndex = 1

  for (let index = 0; index < outputLength; index += 1) {
    const timestamp = startTime + index * intervalMs
    while (
      rightIndex < samples.length - 1 &&
      samples[rightIndex].timestamp < timestamp
    ) {
      rightIndex += 1
    }

    const left = samples[rightIndex - 1]
    const right = samples[rightIndex]
    const ratio =
      (timestamp - left.timestamp) / (right.timestamp - left.timestamp)
    result.push({
      timestamp,
      forehead: interpolateRgb(left.forehead, right.forehead, ratio),
      leftCheek: interpolateRgb(left.leftCheek, right.leftCheek, ratio),
      rightCheek: interpolateRgb(left.rightCheek, right.rightCheek, ratio),
      background: interpolateRgb(left.background, right.background, ratio),
      shoulderY: interpolateNullable(left.shoulderY, right.shoulderY, ratio),
      headVelocity: interpolateNullable(
        left.headVelocity,
        right.headVelocity,
        ratio,
      ),
    })
  }

  return result
}

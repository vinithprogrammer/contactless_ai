import FFT from 'fft.js'

export interface SpectrumPoint {
  frequencyHz: number
  power: number
}

export interface SpectrumResult {
  points: SpectrumPoint[]
  peakFrequencyHz: number | null
  snrDb: number | null
}

interface BiquadCoefficients {
  b0: number
  b1: number
  b2: number
  a1: number
  a2: number
}

const BUTTERWORTH_Q = Math.SQRT1_2

function getLowPassCoefficients(
  cutoffHz: number,
  sampleRate: number,
): BiquadCoefficients {
  const omega = (2 * Math.PI * cutoffHz) / sampleRate
  const cosine = Math.cos(omega)
  const alpha = Math.sin(omega) / (2 * BUTTERWORTH_Q)
  const a0 = 1 + alpha
  return {
    b0: ((1 - cosine) / 2) / a0,
    b1: (1 - cosine) / a0,
    b2: ((1 - cosine) / 2) / a0,
    a1: (-2 * cosine) / a0,
    a2: (1 - alpha) / a0,
  }
}

function getHighPassCoefficients(
  cutoffHz: number,
  sampleRate: number,
): BiquadCoefficients {
  const omega = (2 * Math.PI * cutoffHz) / sampleRate
  const cosine = Math.cos(omega)
  const alpha = Math.sin(omega) / (2 * BUTTERWORTH_Q)
  const a0 = 1 + alpha
  return {
    b0: ((1 + cosine) / 2) / a0,
    b1: (-(1 + cosine)) / a0,
    b2: ((1 + cosine) / 2) / a0,
    a1: (-2 * cosine) / a0,
    a2: (1 - alpha) / a0,
  }
}

function applyBiquad(
  input: readonly number[],
  coefficients: BiquadCoefficients,
): number[] {
  const output = Array.from({ length: input.length }, () => 0)
  let state1 = 0
  let state2 = 0
  for (let index = 0; index < input.length; index += 1) {
    const value = input[index]
    const filtered = coefficients.b0 * value + state1
    state1 =
      coefficients.b1 * value - coefficients.a1 * filtered + state2
    state2 = coefficients.b2 * value - coefficients.a2 * filtered
    output[index] = filtered
  }
  return output
}

function applyCascade(
  input: readonly number[],
  sections: readonly BiquadCoefficients[],
): number[] {
  return sections.reduce(
    (signal, section) => applyBiquad(signal, section),
    [...input],
  )
}

function reflectPad(input: readonly number[], padding: number): number[] {
  const left = Array.from({ length: padding }, (_, index) => input[padding - index])
  const right = Array.from(
    { length: padding },
    (_, index) => input[input.length - 2 - index],
  )
  return [...left, ...input, ...right]
}

function filtfilt(
  input: readonly number[],
  sections: readonly BiquadCoefficients[],
): number[] {
  if (input.length < 3) return [...input]
  const padding = Math.min(input.length - 1, 12)
  const padded = reflectPad(input, padding)
  const forward = applyCascade(padded, sections)
  const reverseFiltered = applyCascade([...forward].reverse(), sections).reverse()
  return reverseFiltered.slice(padding, padding + input.length)
}

export function detrendAndTaper(signal: readonly number[]): number[] {
  const length = signal.length
  if (length === 0) return []

  const meanX = (length - 1) / 2
  const meanY = signal.reduce((sum, value) => sum + value, 0) / length
  let covariance = 0
  let varianceX = 0
  for (let index = 0; index < length; index += 1) {
    const centeredX = index - meanX
    covariance += centeredX * (signal[index] - meanY)
    varianceX += centeredX * centeredX
  }
  const slope = varianceX === 0 ? 0 : covariance / varianceX
  const intercept = meanY - slope * meanX
  const detrended = signal.map((value, index) => value - (intercept + slope * index))
  const edgeLength = Math.max(1, Math.ceil(length * 0.05))

  for (let index = 0; index < edgeLength; index += 1) {
    const position = edgeLength === 1 ? 1 : index / (edgeLength - 1)
    const taper = 0.5 - 0.5 * Math.cos(Math.PI * position)
    detrended[index] *= taper
    detrended[length - index - 1] *= taper
  }
  return detrended
}

export function bandpassZeroPhase(
  signal: readonly number[],
  sampleRate = 30,
  lowCutHz = 0.7,
  highCutHz = 3.5,
): number[] {
  if (
    !Number.isFinite(sampleRate) ||
    sampleRate <= 0 ||
    lowCutHz <= 0 ||
    highCutHz <= lowCutHz ||
    highCutHz >= sampleRate / 2
  ) {
    throw new RangeError('Bandpass cutoffs must be positive and below Nyquist.')
  }
  if (signal.length < 3) return Array.from({ length: signal.length }, () => 0)
  const sections = [
    getHighPassCoefficients(lowCutHz, sampleRate),
    getLowPassCoefficients(highCutHz, sampleRate),
  ]
  return filtfilt(detrendAndTaper(signal), sections)
}

function nextPowerOfTwo(value: number): number {
  let result = 1
  while (result < value) result *= 2
  return result
}

function interpolatePeakFrequency(
  powers: readonly number[],
  peakIndex: number,
  frequencyResolution: number,
): number {
  const left = powers[peakIndex - 1]
  const center = powers[peakIndex]
  const right = powers[peakIndex + 1]
  const denominator = left - 2 * center + right
  const offset =
    peakIndex > 0 &&
    peakIndex < powers.length - 1 &&
    denominator !== 0
      ? 0.5 * (left - right) / denominator
      : 0
  return (peakIndex + Math.max(-0.5, Math.min(0.5, offset))) * frequencyResolution
}

export function computeSpectrum(
  signal: readonly number[],
  sampleRate = 30,
): SpectrumResult {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new RangeError('Spectrum sample rate must be greater than zero.')
  }
  if (signal.length < 4) {
    return { points: [], peakFrequencyHz: null, snrDb: null }
  }

  const fftSize = nextPowerOfTwo(signal.length)
  const fft = new FFT(fftSize)
  const input = Array.from({ length: fftSize }, () => 0)
  const windowSum = signal.length > 1 ? signal.length * 0.54 : 1
  for (let index = 0; index < signal.length; index += 1) {
    const hamming =
      signal.length === 1
        ? 1
        : 0.54 -
          0.46 * Math.cos((2 * Math.PI * index) / (signal.length - 1))
    input[index] = signal[index] * hamming
  }

  const transformed = fft.createComplexArray() as number[]
  fft.realTransform(transformed, input)
  fft.completeSpectrum(transformed)
  const frequencyResolution = sampleRate / fftSize
  const powers = Array.from({ length: fftSize / 2 + 1 }, () => 0)
  for (let bin = 0; bin < powers.length; bin += 1) {
    const real = transformed[2 * bin]
    const imaginary = transformed[2 * bin + 1]
    powers[bin] =
      (real * real + imaginary * imaginary) / (windowSum * windowSum)
  }

  const points = powers.flatMap((power, bin) => {
    const frequencyHz = bin * frequencyResolution
    return frequencyHz >= 0.5 && frequencyHz <= 3.5
      ? [{ frequencyHz, power }]
      : []
  })

  let peakIndex = -1
  for (let bin = 0; bin < powers.length; bin += 1) {
    const frequencyHz = bin * frequencyResolution
    if (
      frequencyHz >= 0.75 &&
      frequencyHz <= 3 &&
      (peakIndex === -1 || powers[bin] > powers[peakIndex])
    ) {
      peakIndex = bin
    }
  }
  if (peakIndex === -1 || powers[peakIndex] <= 0) {
    return { points, peakFrequencyHz: null, snrDb: null }
  }

  const peakFrequencyHz = interpolatePeakFrequency(
    powers,
    peakIndex,
    frequencyResolution,
  )
  let signalPower = 0
  let noisePower = 0
  for (let bin = 0; bin < powers.length; bin += 1) {
    const frequencyHz = bin * frequencyResolution
    if (frequencyHz < 0.7 || frequencyHz > 3.5) continue
    if (Math.abs(frequencyHz - peakFrequencyHz) <= 0.15) {
      signalPower += powers[bin]
    } else {
      noisePower += powers[bin]
    }
  }
  const snrDb =
    signalPower > 0
      ? Math.min(60, 10 * Math.log10(signalPower / Math.max(noisePower, 1e-12)))
      : null
  return { points, peakFrequencyHz, snrDb }
}

export function computePowerSpectrumInBand(
  signal: readonly number[],
  sampleRate: number,
  minFrequencyHz: number,
  maxFrequencyHz: number,
): SpectrumPoint[] {
  if (
    !Number.isFinite(sampleRate) ||
    sampleRate <= 0 ||
    minFrequencyHz < 0 ||
    maxFrequencyHz <= minFrequencyHz ||
    maxFrequencyHz > sampleRate / 2
  ) {
    throw new RangeError('Spectrum band must be valid and below Nyquist.')
  }
  if (signal.length < 4) return []

  const fftSize = nextPowerOfTwo(signal.length)
  const fft = new FFT(fftSize)
  const input = Array.from({ length: fftSize }, () => 0)
  for (let index = 0; index < signal.length; index += 1) {
    const hamming =
      signal.length === 1
        ? 1
        : 0.54 -
          0.46 * Math.cos((2 * Math.PI * index) / (signal.length - 1))
    input[index] = signal[index] * hamming
  }
  const transformed = fft.createComplexArray() as number[]
  fft.realTransform(transformed, input)
  const frequencyResolution = sampleRate / fftSize
  const windowSum = signal.length > 1 ? signal.length * 0.54 : 1
  const points: SpectrumPoint[] = []
  for (let bin = 0; bin <= fftSize / 2; bin += 1) {
    const frequencyHz = bin * frequencyResolution
    if (frequencyHz < minFrequencyHz || frequencyHz > maxFrequencyHz) continue
    const real = transformed[2 * bin]
    const imaginary = transformed[2 * bin + 1]
    points.push({
      frequencyHz,
      power: (real * real + imaginary * imaginary) / (windowSum * windowSum),
    })
  }
  return points
}

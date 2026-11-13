import assert from 'node:assert/strict'
import test from 'node:test'
import { bandpassZeroPhase, computeSpectrum } from '../src/services/dspMath.ts'
import { processPosRegions } from '../src/services/posProcessor.ts'
import {
  calculateFusionWeights,
  fuseRegionalBvp,
} from '../src/services/signalFusion.ts'
import { evaluateSignalQuality } from '../src/services/sqiGatekeeper.ts'
import {
  calculateIbiMetrics,
  detectSystolicPeaks,
  extractVitals,
} from '../src/services/vitalsExtractor.ts'
import {
  resampleUniform,
  SignalBuffer,
} from '../src/utils/ringBuffer.ts'

const sampleRate = 30

function makeSample(timestamp, pulse = 0) {
  return {
    timestamp,
    forehead: { r: 100 + 0.3 * pulse, g: 120 + 1.2 * pulse, b: 110 + 0.15 * pulse },
    leftCheek: { r: 100 + 0.24 * pulse, g: 120 + 0.96 * pulse, b: 110 + 0.12 * pulse },
    rightCheek: { r: 100 + 0.18 * pulse, g: 120 + 0.72 * pulse, b: 110 + 0.09 * pulse },
    background: { r: 70, g: 70, b: 70 },
    shoulderY: 0.65,
    headVelocity: 0.002,
  }
}

test('SignalBuffer rejects absent faces, retains capacity, and clears after a long gap', () => {
  const buffer = new SignalBuffer(4)
  assert.equal(buffer.push(makeSample(0), false, 0), false)
  assert.equal(buffer.length, 0)

  for (let index = 0; index < 5; index += 1) {
    const timestamp = index * 1000 / 30
    assert.equal(buffer.push(makeSample(timestamp), true, timestamp), true)
  }
  assert.equal(buffer.length, 4)
  assert.equal(buffer.getSamples()[0].timestamp, 1000 / 30)

  buffer.push(null, false, 1200)
  assert.equal(buffer.length, 0)
})

test('resampleUniform interpolates onto a 30 Hz grid without extrapolating', () => {
  const samples = [
    makeSample(0, 0),
    makeSample(40, 1),
    makeSample(67, 2),
  ]
  const uniform = resampleUniform(samples, sampleRate, 8)
  assert.equal(uniform.length, 3)
  assert.ok(Math.abs(uniform[1].timestamp - uniform[0].timestamp - 1000 / 30) < 1e-8)
  assert.ok(uniform[0].timestamp >= samples[0].timestamp)
  assert.equal(uniform.at(-1).timestamp, 67)
  assert.ok(uniform[0].forehead.g > 120)
})

test('zero-phase bandpass and FFT recover a known in-band frequency', () => {
  const frequencyHz = 1.25
  const raw = Array.from({ length: 256 }, (_, index) =>
    Math.sin((2 * Math.PI * frequencyHz * index) / sampleRate) + 0.002 * index,
  )
  const filtered = bandpassZeroPhase(raw, sampleRate)
  const spectrum = computeSpectrum(filtered, sampleRate)

  assert.equal(filtered.length, raw.length)
  assert.ok(filtered.every(Number.isFinite))
  assert.ok(Math.abs(spectrum.peakFrequencyHz - frequencyHz) < 0.15)
  assert.ok(Number.isFinite(spectrum.snrDb))
})

test('POS projections recover the pulse frequency independently in all ROIs', () => {
  const frequencyHz = 1.25
  const samples = Array.from({ length: 256 }, (_, index) => {
    const pulse = Math.sin((2 * Math.PI * frequencyHz * index) / sampleRate)
    return makeSample(index * 1000 / sampleRate, pulse)
  })
  const result = processPosRegions(samples, sampleRate)

  for (const region of ['forehead', 'leftCheek', 'rightCheek']) {
    const output = result.regions[region]
    assert.equal(output.filtered.length, samples.length)
    assert.ok(output.filtered.every(Number.isFinite))
    assert.ok(Math.abs(output.spectrum.peakFrequencyHz - frequencyHz) < 0.15)
  }
})

test('regional SNR weights normalize and suppress regions below -3 dB', () => {
  const weights = calculateFusionWeights({
    forehead: 10,
    leftCheek: -6,
    rightCheek: 0,
  })
  assert.ok(Math.abs(weights.forehead + weights.leftCheek + weights.rightCheek - 1) < 1e-12)
  assert.ok(weights.leftCheek < 0.001)
  assert.ok(weights.forehead > weights.rightCheek)

  const fused = fuseRegionalBvp(
    { forehead: 0, leftCheek: 0, rightCheek: 0 },
    {
      forehead: [1, 2],
      leftCheek: [3, 4],
      rightCheek: [5, 6],
    },
  )
  assert.equal(fused.fusedBvp.length, 2)
  assert.ok(Math.abs(fused.percentages.forehead + fused.percentages.leftCheek + fused.percentages.rightCheek - 100) < 1e-10)
  assert.ok(Math.abs(fused.fusedBvp[0] - 3) < 1e-12)
})

test('SQI gate reports no-face, motion, low-light, and ready states', () => {
  const pulse = Array.from({ length: 256 }, (_, index) =>
    Math.sin((2 * Math.PI * 1.2 * index) / sampleRate),
  )
  const base = {
    isFaceValid: true,
    fusedBvp: pulse,
    recentHeadVelocities: [0.001, 0.002, 0.001],
    foreheadMeanRgb: { r: 120, g: 130, b: 110 },
  }
  assert.equal(
    evaluateSignalQuality({ ...base, isFaceValid: false }).state,
    'NO_FACE',
  )
  assert.equal(
    evaluateSignalQuality({
      ...base,
      recentHeadVelocities: [0.001, 0.025],
    }).state,
    'MOTION_ARTIFACT',
  )
  assert.equal(
    evaluateSignalQuality({
      ...base,
      foreheadMeanRgb: { r: 30, g: 30, b: 30 },
    }).state,
    'LOW_LIGHT',
  )
  assert.equal(
    evaluateSignalQuality({
      ...base,
      fusedBvp: [],
      foreheadMeanRgb: { r: 40.1, g: 40.1, b: 40.1 },
    }).state,
    'LOW_SQI',
  )
  assert.equal(evaluateSignalQuality(base).state, 'READY')
})

test('RMSSD needs four consecutive inlier intervals and does not bridge an outlier', () => {
  const consecutive = calculateIbiMetrics([0, 30, 61, 90, 120])
  assert.ok(consecutive.rmssdMs !== null)

  const withOutlierGap = calculateIbiMetrics([0, 30, 60, 90, 150, 180, 210, 240])
  assert.equal(withOutlierGap.rmssdMs, null)
})

test('vitals extractor returns peak, HRV, respiratory, and bounded SpO2 proxy', () => {
  const frequencyHz = 1.2
  const samples = Array.from({ length: 256 }, (_, index) => {
    const time = index / sampleRate
    const pulse = Math.sin(2 * Math.PI * frequencyHz * time)
    const breathing = Math.sin(2 * Math.PI * 0.25 * time)
    return {
      timestamp: index * 1000 / sampleRate,
      forehead: {
        r: 120 + 1.2 * pulse,
        g: 130 + pulse,
        b: 110 + 0.8 * pulse,
      },
      leftCheek: makeSample(index).leftCheek,
      rightCheek: makeSample(index).rightCheek,
      background: { r: 70, g: 70, b: 70 },
      shoulderY: 0.6 + 0.01 * breathing,
      headVelocity: 0.001,
    }
  })
  const signal = Array.from({ length: samples.length }, (_, index) =>
    (1 + 0.1 * Math.sin((2 * Math.PI * 0.25 * index) / sampleRate)) *
      Math.sin((2 * Math.PI * frequencyHz * index) / sampleRate),
  )
  const fusion = fuseRegionalBvp(
    { forehead: 12, leftCheek: 8, rightCheek: 6 },
    { forehead: signal, leftCheek: signal, rightCheek: signal },
  )
  const peaks = detectSystolicPeaks(signal)
  const ibi = calculateIbiMetrics(peaks)
  const vitals = extractVitals(samples, fusion, null)

  assert.ok(peaks.length >= 8)
  assert.ok(Math.abs(vitals.heartRateBpm - 72) < 2)
  assert.equal(ibi.rmssdMs, 0)
  assert.equal(vitals.autonomicStress, 'Elevated Sympathetic Stress')
  assert.ok(Math.abs(vitals.respiratoryRateBrpm - 15) <= 2)
  assert.ok(vitals.spo2ProxyPercent >= 88 && vitals.spo2ProxyPercent <= 100)
  assert.ok(vitals.peakIndices.length > 0)

  const opticalOnlySamples = samples.map((sample) => ({
    ...sample,
    shoulderY: 0.6,
  }))
  const opticalOnlyVitals = extractVitals(opticalOnlySamples, fusion, null)
  assert.ok(Math.abs(opticalOnlyVitals.respiratoryRateBrpm - 15) <= 2)
})


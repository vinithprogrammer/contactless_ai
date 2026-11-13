import { useEffect, useRef, useState } from 'react'
import {
  Activity,
  Camera,
  CheckCircle2,
  CircleHelp,
  Heart,
  HeartPulse,
  Info,
  ScanFace,
  Signal,
  Video,
} from 'lucide-react'
import { initializeVisionEngine, type VisionFrame } from './services/visionEngine'
import {
  processPosRegions,
  type PosProcessingResult,
  type ProcessedRegion,
} from './services/posProcessor'
import {
  fuseRegionalBvp,
  type SignalFusionResult,
} from './services/signalFusion'
import {
  evaluateSignalQuality,
  type SqiGateResult,
} from './services/sqiGatekeeper'
import {
  DEFAULT_OPTICAL_CALIBRATION,
  extractVitals,
  MIN_VITAL_SAMPLE_COUNT,
  type ExtractedVitals,
  type OpticalCalibration,
} from './services/vitalsExtractor'
import {
  BACKGROUND_ROI,
  ROI_LANDMARKS,
  type Landmark,
  type PixelRegion,
  type RGB,
  type RoiExtraction,
} from './utils/roiExtractor'
import {
  resampleUniform,
  SIGNAL_BUFFER_CAPACITY,
  SIGNAL_BUFFER_WARMUP,
  SignalBuffer,
  type SignalSample,
} from './utils/ringBuffer'

interface Telemetry {
  extraction: RoiExtraction | null
  fps: number
  width: number
  height: number
  bufferLength: number
  warmupLength: number
  processing: PosProcessingResult | null
  fusion: SignalFusionResult | null
  sqi: SqiGateResult
  vitals: ExtractedVitals
}

interface ChartState {
  faceValid: boolean
  warmupLength: number
  processing: PosProcessingResult | null
  fusion: SignalFusionResult | null
  peakIndices: number[]
}

const EMPTY_REGION: PixelRegion = { count: 0, mean: null }

const EMPTY_EXTRACTION: RoiExtraction = {
  isFaceValid: false,
  rgbSample: null,
  forehead: EMPTY_REGION,
  leftCheek: EMPTY_REGION,
  rightCheek: EMPTY_REGION,
  background: EMPTY_REGION,
  headVelocity: null,
  shoulderY: null,
  faceLandmarks: null,
}

const CHART_WIDTH = 900
const CHART_HEIGHT = 260
const EMPTY_VITALS: ExtractedVitals = {
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
const NO_FACE_SQI = evaluateSignalQuality({
  isFaceValid: false,
  fusedBvp: [],
  recentHeadVelocities: [],
  foreheadMeanRgb: null,
})
function drawPolygon(
  context: CanvasRenderingContext2D,
  landmarks: Landmark[],
  indices: readonly number[],
  color: string,
  width: number,
  height: number,
) {
  const points = indices
    .map((index) => landmarks[index])
    .filter((point): point is Landmark => Boolean(point))
  if (points.length !== indices.length) return

  context.beginPath()
  points.forEach((point, index) => {
    const x = point.x * width
    const y = point.y * height
    if (index === 0) context.moveTo(x, y)
    else context.lineTo(x, y)
  })
  context.closePath()
  context.fillStyle = color
  context.globalAlpha = 0.14
  context.fill()
  context.globalAlpha = 1
  context.strokeStyle = color
  context.lineWidth = Math.max(2, width / 480)
  context.stroke()
}

function drawDiagnosticsOverlay(
  canvas: HTMLCanvasElement,
  frame: VisionFrame,
) {
  if (canvas.width !== frame.width || canvas.height !== frame.height) {
    canvas.width = frame.width
    canvas.height = frame.height
  }
  const context = canvas.getContext('2d')
  if (!context) return

  context.clearRect(0, 0, frame.width, frame.height)
  const landmarks = frame.extraction.faceLandmarks
  if (landmarks) {
    drawPolygon(
      context,
      landmarks,
      ROI_LANDMARKS.forehead,
      '#22d3ee',
      frame.width,
      frame.height,
    )
    drawPolygon(
      context,
      landmarks,
      ROI_LANDMARKS.leftCheek,
      '#34d399',
      frame.width,
      frame.height,
    )
    drawPolygon(
      context,
      landmarks,
      ROI_LANDMARKS.rightCheek,
      '#a78bfa',
      frame.width,
      frame.height,
    )
  }

  context.strokeStyle = '#fbbf24'
  context.lineWidth = Math.max(2, frame.width / 480)
  context.setLineDash([8, 6])
  context.strokeRect(
    BACKGROUND_ROI.x,
    BACKGROUND_ROI.y,
    BACKGROUND_ROI.width,
    BACKGROUND_ROI.height,
  )
  context.setLineDash([])
}

function drawChartHeader(
  context: CanvasRenderingContext2D,
  title: string,
  width: number,
): void {
  context.fillStyle = '#0f172a'
  context.fillRect(0, 0, width, 28)
  context.fillStyle = '#cbd5e1'
  context.font = '600 12px ui-sans-serif, system-ui, sans-serif'
  context.textBaseline = 'middle'
  context.fillText(title, 14, 14)
}

function getChartStatus(state: ChartState): string | null {
  if (!state.faceValid) return 'NO FACE DETECTED — SIGNAL PAUSED'
  if (state.warmupLength < SIGNAL_BUFFER_WARMUP) {
    return `ACQUIRING SIGNAL BUFFER (${state.warmupLength}/${SIGNAL_BUFFER_WARMUP} FRAMES)...`
  }
  return null
}

function drawFlatline(
  context: CanvasRenderingContext2D,
  area: { left: number; top: number; right: number; bottom: number },
  message: string,
  zeroAtBottom = false,
): void {
  const y = zeroAtBottom ? area.bottom : (area.top + area.bottom) / 2
  context.strokeStyle = '#475569'
  context.lineWidth = 1
  context.beginPath()
  context.moveTo(area.left, y)
  context.lineTo(area.right, y)
  context.stroke()
  context.fillStyle = '#94a3b8'
  context.font = '500 12px ui-sans-serif, system-ui, sans-serif'
  context.textAlign = 'center'
  context.textBaseline = 'middle'
  context.fillText(message, (area.left + area.right) / 2, (area.top + area.bottom) / 2)
  context.textAlign = 'start'
}

function drawGrid(
  context: CanvasRenderingContext2D,
  area: { left: number; top: number; right: number; bottom: number },
): void {
  context.strokeStyle = 'rgba(71, 85, 105, 0.28)'
  context.lineWidth = 1
  for (let step = 0; step <= 4; step += 1) {
    const y = area.top + ((area.bottom - area.top) * step) / 4
    context.beginPath()
    context.moveTo(area.left, y)
    context.lineTo(area.right, y)
    context.stroke()
  }
}

function drawSeries(
  context: CanvasRenderingContext2D,
  values: readonly number[],
  area: { left: number; top: number; right: number; bottom: number },
  color: string,
  lineWidth: number,
  alpha = 1,
  markers: readonly number[] = [],
  markerColor = '#34d399',
): void {
  if (values.length < 2) return
  const center =
    values.reduce((sum, value) => sum + value, 0) / values.length
  const amplitude = Math.max(
    ...values.map((value) => Math.abs(value - center)),
    1e-8,
  )
  const verticalScale = ((area.bottom - area.top) * 0.43) / amplitude
  context.strokeStyle = color
  context.lineWidth = lineWidth
  context.globalAlpha = alpha
  context.beginPath()
  values.forEach((value, index) => {
    const x = area.left + (index / (values.length - 1)) * (area.right - area.left)
    const y = (area.top + area.bottom) / 2 - (value - center) * verticalScale
    if (index === 0) context.moveTo(x, y)
    else context.lineTo(x, y)
  })
  context.stroke()
  if (markers.length > 0) {
    context.fillStyle = markerColor
    for (const index of markers) {
      if (index < 0 || index >= values.length) continue
      const x =
        area.left + (index / (values.length - 1)) * (area.right - area.left)
      const y = (area.top + area.bottom) / 2 - (values[index] - center) * verticalScale
      context.beginPath()
      context.arc(x, y, 3.5, 0, 2 * Math.PI)
      context.fill()
    }
  }
  context.globalAlpha = 1
}

function drawBvpPlot(canvas: HTMLCanvasElement, state: ChartState): void {
  const context = canvas.getContext('2d')
  if (!context) return
  const width = canvas.width
  const height = canvas.height
  context.clearRect(0, 0, width, height)
  context.fillStyle = '#020617'
  context.fillRect(0, 0, width, height)
  drawChartHeader(context, 'BVP Waveform', width)
  if (state.fusion) {
    const weights = state.fusion.percentages
    context.fillStyle = '#94a3b8'
    context.font = '10px ui-monospace, monospace'
    context.textAlign = 'right'
    context.textBaseline = 'middle'
    context.fillText(
      `w1 F ${weights.forehead.toFixed(0)}% · w2 L ${weights.leftCheek.toFixed(0)}% · w3 R ${weights.rightCheek.toFixed(0)}%`,
      width - 14,
      14,
    )
    context.textAlign = 'start'
  }
  const area = { left: 44, top: 50, right: width - 18, bottom: height - 24 }
  drawGrid(context, area)

  const status = getChartStatus(state)
  const fusedBvp = state.fusion?.fusedBvp
  if (!status && fusedBvp && fusedBvp.length > 1) {
    drawSeries(context, state.processing?.rawGreen ?? [], area, '#34d399', 1.5, 0.55)
    drawSeries(
      context,
      fusedBvp,
      area,
      '#22d3ee',
      2,
      1,
      state.peakIndices,
    )
  } else {
    drawFlatline(
      context,
      area,
      status ?? 'SIGNAL PAUSED',
    )
  }

  context.font = '10px ui-sans-serif, system-ui, sans-serif'
  context.textBaseline = 'alphabetic'
  context.fillStyle = '#34d399'
  context.fillText('Normalized green', area.left, height - 7)
  context.fillStyle = '#22d3ee'
  context.fillText('Fused BVP · systolic peaks', area.left + 112, height - 7)
}

function drawSpectrumPlot(canvas: HTMLCanvasElement, state: ChartState): void {
  const context = canvas.getContext('2d')
  if (!context) return
  const width = canvas.width
  const height = canvas.height
  context.clearRect(0, 0, width, height)
  context.fillStyle = '#020617'
  context.fillRect(0, 0, width, height)
  drawChartHeader(context, 'FFT Spectrum (0.5–3.5 Hz)', width)
  const peakFrequency = state.processing?.regions.forehead.spectrum.peakFrequencyHz
  if (peakFrequency !== null && peakFrequency !== undefined) {
    context.fillStyle = '#67e8f9'
    context.font = '600 11px ui-monospace, monospace'
    context.textAlign = 'right'
    context.textBaseline = 'middle'
    context.fillText(`${peakFrequency.toFixed(2)} Hz peak`, width - 14, 42)
    context.textAlign = 'start'
  }

  const area = { left: 44, top: 57, right: width - 18, bottom: height - 30 }
  drawGrid(context, area)
  const status = getChartStatus(state)
  const points = state.processing?.regions.forehead.spectrum.points ?? []
  if (!status && points.length > 1) {
    const maxPower = Math.max(...points.map(({ power }) => power), 1e-12)
    const firstFrequency = 0.5
    const frequencySpan = 3
    context.strokeStyle = '#22d3ee'
    context.lineWidth = 2
    context.beginPath()
    points.forEach((point, index) => {
      const x =
        area.left +
        ((point.frequencyHz - firstFrequency) / frequencySpan) *
          (area.right - area.left)
      const y =
        area.bottom -
        (point.power / maxPower) * (area.bottom - area.top) * 0.88
      if (index === 0) context.moveTo(x, y)
      else context.lineTo(x, y)
    })
    context.stroke()
  } else {
    drawFlatline(context, area, status ?? 'SPECTRUM UNAVAILABLE', true)
  }

  context.fillStyle = '#64748b'
  context.font = '10px ui-monospace, monospace'
  context.textBaseline = 'alphabetic'
  context.fillText('0.5 Hz', area.left, height - 7)
  context.textAlign = 'right'
  context.fillText('3.5 Hz', area.right, height - 7)
  context.textAlign = 'start'
}

function formatRgb(value: RGB | null | undefined): string {
  if (!value) return '--'
  return `${Math.round(value.r)}, ${Math.round(value.g)}, ${Math.round(value.b)}`
}

function formatSnr(region: ProcessedRegion | undefined): string {
  const snr = region?.spectrum.snrDb
  return snr === null || snr === undefined ? '-- dB' : `${snr.toFixed(1)} dB`
}

function RegionRow({
  label,
  region,
  color,
  average,
}: {
  label: string
  region: PixelRegion
  color: string
  average: RGB | null | undefined
}) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_4.5rem_7.5rem] items-center gap-3 border-b border-slate-800/80 py-2.5 last:border-0">
      <div className="flex min-w-0 items-center gap-2.5">
        <span className={`h-2 w-2 shrink-0 rounded-full ${color}`} />
        <span className="truncate text-xs text-slate-300">{label}</span>
      </div>
      <span className="text-right font-mono text-[11px] tabular-nums text-slate-400">
        {region.count}
      </span>
      <span className="text-right font-mono text-[11px] tabular-nums text-slate-200">
        {formatRgb(average)}
      </span>
    </div>
  )
}

function SnrBadge({
  label,
  region,
  color,
}: {
  label: string
  region: ProcessedRegion | undefined
  color: string
}) {
  return (
    <div className="flex min-w-0 flex-1 items-center justify-between gap-2 rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2">
      <span className={`truncate text-[11px] font-medium ${color}`}>{label}</span>
      <span className="whitespace-nowrap font-mono text-xs tabular-nums text-slate-200">
        {formatSnr(region)}
      </span>
    </div>
  )
}

function retainValidVitals(
  previous: ExtractedVitals,
  next: ExtractedVitals,
): ExtractedVitals {
  return {
    heartRateBpm: next.heartRateBpm ?? previous.heartRateBpm,
    spectralHeartRateBpm:
      next.spectralHeartRateBpm ?? previous.spectralHeartRateBpm,
    ibiHeartRateBpm: next.ibiHeartRateBpm ?? previous.ibiHeartRateBpm,
    peakIndices: next.peakIndices,
    rmssdMs: next.rmssdMs ?? previous.rmssdMs,
    autonomicStress: next.autonomicStress ?? previous.autonomicStress,
    respiratoryRateBrpm:
      next.respiratoryRateBrpm ?? previous.respiratoryRateBrpm,
    spo2ProxyPercent: next.spo2ProxyPercent ?? previous.spo2ProxyPercent,
    rRatio: next.rRatio ?? previous.rRatio,
    ibiConsistencyPercent:
      next.ibiConsistencyPercent ?? previous.ibiConsistencyPercent,
    confidencePercent: next.confidencePercent ?? previous.confidencePercent,
  }
}

function formatVital(value: number | null, digits = 0): string {
  return value === null ? '--' : value.toFixed(digits)
}

function stressColor(stress: ExtractedVitals['autonomicStress']): string {
  if (stress === 'Optimal / Parasympathetic Recovery') {
    return 'border-emerald-400/20 bg-emerald-400/10 text-emerald-300'
  }
  if (stress === 'Balanced / Normal Tone') {
    return 'border-cyan-400/20 bg-cyan-400/10 text-cyan-300'
  }
  if (stress === 'Elevated Sympathetic Stress') {
    return 'border-amber-400/20 bg-amber-400/10 text-amber-200'
  }
  return 'border-slate-700 bg-slate-800 text-slate-400'
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const waveformRef = useRef<HTMLCanvasElement>(null)
  const spectrumRef = useRef<HTMLCanvasElement>(null)
  const previousBpmRef = useRef<number | null>(null)
  const lastValidVitalsRef = useRef<ExtractedVitals>(EMPTY_VITALS)
  const calibrationRef = useRef<OpticalCalibration>({
    ...DEFAULT_OPTICAL_CALIBRATION,
  })
  const [calibration, setCalibration] = useState<OpticalCalibration>({
    ...DEFAULT_OPTICAL_CALIBRATION,
  })
  const [telemetry, setTelemetry] = useState<Telemetry>({
    extraction: null,
    fps: 0,
    width: 0,
    height: 0,
    bufferLength: 0,
    warmupLength: 0,
    processing: null,
    fusion: null,
    sqi: NO_FACE_SQI,
    vitals: EMPTY_VITALS,
  })
  const [cameraStatus, setCameraStatus] = useState<
    'initializing' | 'ready' | 'error'
  >('initializing')
  const [cameraError, setCameraError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    let stream: MediaStream | null = null
    let stopPipeline: (() => void) | null = null
    let engine: Awaited<ReturnType<typeof initializeVisionEngine>> | null = null
    let previousTimestamp = 0
    let lastTelemetryUpdate = 0
    const signalBuffer = new SignalBuffer(SIGNAL_BUFFER_CAPACITY)
    const initialChartState: ChartState = {
      faceValid: false,
      warmupLength: 0,
      processing: null,
      fusion: null,
      peakIndices: [],
    }
    if (waveformRef.current) drawBvpPlot(waveformRef.current, initialChartState)
    if (spectrumRef.current) {
      drawSpectrumPlot(spectrumRef.current, initialChartState)
    }

    const startCamera = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error('This browser does not support camera access.')
        }
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: 'user' },
        })
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop())
          return
        }

        const video = videoRef.current
        const overlay = overlayRef.current
        if (!video || !overlay) {
          throw new Error('The diagnostic video surface is unavailable.')
        }
        video.srcObject = stream
        await video.play()

        engine = await initializeVisionEngine()
        if (cancelled) {
          engine.close()
          return
        }

        setCameraStatus('ready')
        const processingCanvas = document.createElement('canvas')
        stopPipeline = engine.start(
          video,
          processingCanvas,
          (frame) => {
            drawDiagnosticsOverlay(overlay, frame)
            const sample = makeSignalSample(frame)
            signalBuffer.push(
              sample,
              frame.extraction.isFaceValid,
              frame.timestampMs,
            )
            const bufferLength = signalBuffer.length
            const uniformSamples =
              frame.extraction.isFaceValid && signalBuffer.ready
                ? resampleUniform(
                    signalBuffer.getSamples(),
                    30,
                    SIGNAL_BUFFER_CAPACITY,
                  )
                : []
            const warmupLength = signalBuffer.ready
              ? Math.min(bufferLength, uniformSamples.length)
              : bufferLength
            const processing =
              frame.extraction.isFaceValid &&
              warmupLength >= SIGNAL_BUFFER_WARMUP
                ? processPosRegions(uniformSamples, 30)
                : null
            const fusion = processing
              ? fuseRegionalBvp(
                  {
                    forehead: processing.regions.forehead.spectrum.snrDb,
                    leftCheek: processing.regions.leftCheek.spectrum.snrDb,
                    rightCheek: processing.regions.rightCheek.spectrum.snrDb,
                  },
                  {
                    forehead: processing.regions.forehead.filtered,
                    leftCheek: processing.regions.leftCheek.filtered,
                    rightCheek: processing.regions.rightCheek.filtered,
                  },
                )
              : null
            const currentSamples = signalBuffer.getSamples()
            const sqi = evaluateSignalQuality({
              isFaceValid: frame.extraction.isFaceValid,
              fusedBvp: fusion?.fusedBvp ?? [],
              recentHeadVelocities: currentSamples
                .slice(-30)
                .map((sample) => sample.headVelocity),
              foreheadMeanRgb: frame.extraction.rgbSample?.forehead ?? null,
            })
            let frameVitals: ExtractedVitals | null = null
            if (
              sqi.state === 'READY' &&
              warmupLength >= MIN_VITAL_SAMPLE_COUNT &&
              fusion
            ) {
              frameVitals = extractVitals(
                uniformSamples,
                fusion,
                previousBpmRef.current,
                calibrationRef.current,
              )
              if (frameVitals.heartRateBpm !== null) {
                previousBpmRef.current = frameVitals.heartRateBpm
              }
              lastValidVitalsRef.current = retainValidVitals(
                lastValidVitalsRef.current,
                frameVitals,
              )
            }
            const chartState = {
              faceValid: frame.extraction.isFaceValid,
              warmupLength,
              processing,
              fusion,
              peakIndices: frameVitals?.peakIndices ?? [],
            }
            if (waveformRef.current) {
              drawBvpPlot(waveformRef.current, chartState)
            }
            if (spectrumRef.current) {
              drawSpectrumPlot(spectrumRef.current, chartState)
            }

            const elapsed = frame.timestampMs - previousTimestamp
            previousTimestamp = frame.timestampMs
            if (
              !frame.extraction.isFaceValid ||
              warmupLength < SIGNAL_BUFFER_WARMUP ||
              frame.timestampMs - lastTelemetryUpdate >= 125
            ) {
              setTelemetry({
                extraction: frame.extraction,
                fps: elapsed > 0 ? 1000 / elapsed : 0,
                width: frame.width,
                height: frame.height,
                bufferLength,
                warmupLength,
                processing,
                fusion,
                sqi,
                vitals: lastValidVitalsRef.current,
              })
              lastTelemetryUpdate = frame.timestampMs
            }
          },
          (error) => {
            if (cancelled) return
            setCameraStatus('error')
            setCameraError(
              error instanceof Error ? error.message : String(error),
            )
            stream?.getTracks().forEach((track) => track.stop())
          },
        )
      } catch (error) {
        if (cancelled) return
        engine?.close()
        stream?.getTracks().forEach((track) => track.stop())
        setCameraStatus('error')
        setCameraError(error instanceof Error ? error.message : String(error))
      }
    }

    void startCamera()
    return () => {
      cancelled = true
      stopPipeline?.()
      engine?.close()
      stream?.getTracks().forEach((track) => track.stop())
    }
  }, [])

  const extraction = telemetry.extraction ?? EMPTY_EXTRACTION
  const faceValid = extraction.isFaceValid
  const processing = faceValid ? telemetry.processing : null
  const regions = processing?.regions
  const vitals = telemetry.vitals
  const isSqiReady = telemetry.sqi.state === 'READY'

  const updateCalibration = (field: keyof OpticalCalibration, value: number) => {
    const updated = { ...calibrationRef.current, [field]: value }
    calibrationRef.current = updated
    setCalibration(updated)
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-[1560px] px-5 py-6 sm:px-8 lg:px-10">
        <header className="mb-8 flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-6">
          <div className="flex items-center gap-3">
            <div className="grid h-11 w-11 place-items-center rounded-xl border border-cyan-400/20 bg-cyan-400/10 text-cyan-300">
              <Activity size={21} strokeWidth={1.8} />
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight text-slate-100">
                VitalsLens<span className="text-cyan-300">-AI</span>
              </h1>
              <p className="mt-0.5 text-xs tracking-wide text-slate-500">
                CONTACTLESS VISION · DSP ACQUISITION
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/70 px-3 py-1.5">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                cameraStatus === 'ready'
                  ? 'bg-emerald-400'
                  : cameraStatus === 'error'
                    ? 'bg-rose-400'
                    : 'animate-pulse bg-amber-400'
              }`}
            />
            <span className="text-xs font-medium text-slate-400">
              {cameraStatus === 'ready'
                ? 'VISION ENGINE ACTIVE'
                : cameraStatus === 'error'
                  ? 'CAMERA / MODEL ERROR'
                  : 'INITIALIZING VISION ENGINE'}
            </span>
          </div>
        </header>

        <div className="mb-5 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.16em] text-slate-500">
          <ScanFace size={15} className="text-cyan-400" />
          <span>ROI acquisition</span>
          <span className="text-slate-700">/</span>
          <span>POS projection</span>
          <span className="text-slate-700">/</span>
          <span>Zero-phase DSP</span>
        </div>

        <div className="grid gap-5 xl:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
          <div className="space-y-5">
            <section className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/55 shadow-2xl shadow-black/20">
              <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
                <div className="flex items-center gap-2.5">
                  <Video size={16} className="text-slate-400" />
                  <h2 className="text-sm font-semibold text-slate-200">
                    Live camera feed
                  </h2>
                </div>
                <span className="font-mono text-[11px] text-slate-500">
                  {telemetry.width > 0
                    ? `${telemetry.width} × ${telemetry.height}`
                    : 'AWAITING SOURCE'}
                </span>
              </div>
              <div
                className="relative mx-4 my-4 overflow-hidden rounded-xl bg-slate-950"
                style={{
                  aspectRatio:
                    telemetry.width > 0
                      ? `${telemetry.width} / ${telemetry.height}`
                      : '16 / 9',
                }}
              >
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  className="absolute inset-0 h-full w-full object-fill"
                />
                <canvas
                  ref={overlayRef}
                  className="pointer-events-none absolute inset-0 h-full w-full"
                />
                {!faceValid && (
                  <div className="absolute inset-x-0 top-0 z-10 flex min-h-12 items-center justify-center gap-2 border-b border-amber-300/30 bg-amber-400/15 px-3 py-3 text-center text-xs font-bold tracking-[0.08em] text-amber-200 backdrop-blur-sm sm:text-sm">
                    NO FACE DETECTED — SIGNAL PAUSED
                  </div>
                )}
                {cameraStatus === 'initializing' && (
                  <div className="absolute inset-0 grid place-items-center bg-slate-950/40">
                    <div className="flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900/90 px-4 py-3 text-sm text-slate-300">
                      <Camera size={16} className="text-cyan-300" />
                      Requesting camera and loading vision models
                    </div>
                  </div>
                )}
                {cameraStatus === 'error' && (
                  <div className="absolute inset-x-4 bottom-4 rounded-lg border border-rose-400/30 bg-rose-950/90 p-3 text-sm text-rose-200">
                    <p className="font-semibold">Unable to start diagnostics</p>
                    <p className="mt-1 break-words text-xs text-rose-200/80">
                      {cameraError}
                    </p>
                  </div>
                )}
              </div>
              <div className="flex flex-wrap gap-x-5 gap-y-2 border-t border-slate-800 px-5 py-3.5">
                <Legend color="bg-cyan-400" label="Forehead" />
                <Legend color="bg-emerald-400" label="Left cheek" />
                <Legend color="bg-violet-400" label="Right cheek" />
                <Legend color="bg-amber-400" label="Illumination reference" />
              </div>
            </section>

            <section className="rounded-2xl border border-slate-800 bg-slate-900/55">
              <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
                <div className="flex items-center gap-2.5">
                  <Signal size={16} className="text-cyan-300" />
                  <h2 className="text-sm font-semibold text-slate-200">
                    Spatial ROI samples
                  </h2>
                </div>
                <span
                  className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-bold tracking-wider ${
                    faceValid
                      ? 'border-emerald-400/20 bg-emerald-400/10 text-emerald-300'
                      : 'border-amber-400/20 bg-amber-400/10 text-amber-200'
                  }`}
                >
                  {faceValid ? (
                    <CheckCircle2 size={12} />
                  ) : (
                    <CircleHelp size={12} />
                  )}
                  {faceValid ? 'FACE LOCKED' : 'WAITING'}
                </span>
              </div>
              <div className="px-5 pb-2">
                <div className="grid grid-cols-[minmax(0,1fr)_4.5rem_7.5rem] gap-3 border-b border-slate-800 pb-2 pt-3 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                  <span>Region</span>
                  <span className="text-right">Pixels</span>
                  <span className="text-right">Mean RGB</span>
                </div>
                <RegionRow
                  label="Forehead"
                  region={extraction.forehead}
                  average={faceValid ? extraction.rgbSample?.forehead : null}
                  color="bg-cyan-400"
                />
                <RegionRow
                  label="Left cheek"
                  region={extraction.leftCheek}
                  average={faceValid ? extraction.rgbSample?.leftCheek : null}
                  color="bg-emerald-400"
                />
                <RegionRow
                  label="Right cheek"
                  region={extraction.rightCheek}
                  average={faceValid ? extraction.rgbSample?.rightCheek : null}
                  color="bg-violet-400"
                />
                <RegionRow
                  label="Background reference"
                  region={extraction.background}
                  average={faceValid ? extraction.rgbSample?.background : null}
                  color="bg-amber-400"
                />
              </div>
              <div className="flex flex-wrap justify-between gap-2 border-t border-slate-800 px-5 py-3 text-[10px] uppercase tracking-wider text-slate-500">
                <span>Processing {telemetry.fps.toFixed(1)} FPS</span>
                <span>
                  Buffer {telemetry.bufferLength}/{SIGNAL_BUFFER_CAPACITY}
                </span>
                <span>Warm-up {SIGNAL_BUFFER_WARMUP} valid frames</span>
              </div>
            </section>
          </div>

          <div className="space-y-5">
            <section className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/55 shadow-xl shadow-black/10">
              <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
                <div className="flex items-center gap-2.5">
                  <Activity size={16} className="text-cyan-300" />
                  <h2 className="text-sm font-semibold text-slate-200">
                    Live signal processing
                  </h2>
                </div>
                <span className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
                  Uniform 30 Hz · POS L=48 · 0.7–3.5 Hz
                </span>
              </div>
              <div className="p-3 sm:p-4">
                <canvas
                  ref={waveformRef}
                  width={CHART_WIDTH}
                  height={CHART_HEIGHT}
                  className="h-[210px] w-full rounded-lg border border-slate-800 bg-slate-950"
                  aria-label="Normalized green channel and master fused BVP waveform with systolic peak markers"
                />
                <div className="mt-2 flex items-center justify-between px-1 text-[10px] text-slate-500">
                  <span>Normalized green vs. master fused BVP</span>
                  <span>
                    {telemetry.warmupLength >= SIGNAL_BUFFER_WARMUP
                      ? 'DSP ACTIVE'
                      : 'BUFFER WARM-UP'}
                  </span>
                </div>
              </div>
            </section>

            <section className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/55 shadow-xl shadow-black/10">
              <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
                <div className="flex items-center gap-2.5">
                  <Activity size={16} className="text-violet-300" />
                  <h2 className="text-sm font-semibold text-slate-200">
                    Regional frequency spectra
                  </h2>
                </div>
                <span className="text-[10px] uppercase tracking-wider text-slate-500">
                  Hamming window · PSD
                </span>
              </div>
              <div className="p-3 sm:p-4">
                <canvas
                  ref={spectrumRef}
                  width={CHART_WIDTH}
                  height={CHART_HEIGHT}
                  className="h-[210px] w-full rounded-lg border border-slate-800 bg-slate-950"
                  aria-label="Forehead FFT power spectrum from 0.5 to 3.5 hertz"
                />
                <div className="mt-3 flex flex-wrap gap-2">
                  <SnrBadge
                    label="Forehead"
                    region={regions?.forehead}
                    color="text-cyan-300"
                  />
                  <SnrBadge
                    label="L-Cheek"
                    region={regions?.leftCheek}
                    color="text-emerald-300"
                  />
                  <SnrBadge
                    label="R-Cheek"
                    region={regions?.rightCheek}
                    color="text-violet-300"
                  />
                </div>
              </div>
              <p className="border-t border-slate-800 px-5 py-3 text-[10px] leading-relaxed text-slate-600">
                SNR is peak-band power (±0.15 Hz) relative to remaining 0.7–3.5
                Hz band power. Readouts are signal-processing diagnostics, not
                clinical measurements.
              </p>
            </section>
          </div>
        </div>

        <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/55 p-5 shadow-xl shadow-black/10">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Signal size={17} className="text-cyan-300" />
              <h2 className="text-sm font-semibold text-slate-200">
                Signal quality
              </h2>
            </div>
            <span className="font-mono text-sm tabular-nums text-slate-200">
              {telemetry.sqi.sqiScore}%
            </span>
          </div>
          <div
            className="h-2 overflow-hidden rounded-full bg-slate-800"
            role="progressbar"
            aria-label="Signal quality index"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={telemetry.sqi.sqiScore}
          >
            <div
              className={`h-full rounded-full transition-[width] duration-300 ${
                telemetry.sqi.sqiScore >= 70
                  ? 'bg-emerald-400'
                  : telemetry.sqi.sqiScore >= 35
                    ? 'bg-cyan-400'
                    : 'bg-amber-400'
              }`}
              style={{ width: `${telemetry.sqi.sqiScore}%` }}
            />
          </div>
          <div
            className={`mt-4 flex items-center gap-2 rounded-lg border px-3 py-2.5 text-xs font-semibold tracking-wide ${
              isSqiReady
                ? 'border-emerald-400/20 bg-emerald-400/5 text-emerald-300'
                : 'border-amber-400/20 bg-amber-400/5 text-amber-200'
            }`}
            role={isSqiReady ? 'status' : 'alert'}
          >
            {isSqiReady ? <CheckCircle2 size={15} /> : <Info size={15} />}
            {telemetry.sqi.message}
          </div>
          <p className="mt-2 text-[10px] text-slate-500">
            SQI combines cardiac-band spectral concentration (60%), recent head
            stability (25%), and forehead illumination (15%).
          </p>
        </section>

        <section className="mt-6">
          <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
            <div>
              <h2 className="text-base font-semibold text-slate-100">
                Clinical telemetry
                <span className="ml-2 text-xs font-medium text-cyan-300">
                  Research estimates
                </span>
              </h2>
              <p className="mt-1 text-xs text-slate-500">
                Requires {MIN_VITAL_SAMPLE_COUNT} valid frames and acceptable
                SQI. Values hold during signal alerts.
              </p>
            </div>
            <span className="text-[10px] uppercase tracking-wider text-slate-600">
              Not for diagnosis or treatment decisions
            </span>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <article className="rounded-2xl border border-slate-800 bg-slate-900/55 p-5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Heart rate
                </span>
                <Heart size={16} className="text-rose-300" />
              </div>
              <div className="mt-4 flex items-baseline gap-2">
                <span className="font-mono text-4xl font-semibold tabular-nums text-slate-100">
                  {formatVital(vitals.heartRateBpm)}
                </span>
                <span className="text-sm text-slate-500">BPM</span>
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                <ReadoutBadge
                  label={`Confidence ${formatVital(vitals.confidencePercent)}%`}
                  tone="cyan"
                />
                <ReadoutBadge
                  label={`IBI stable ${formatVital(vitals.ibiConsistencyPercent)}%`}
                  tone="slate"
                />
              </div>
            </article>

            <article className="rounded-2xl border border-slate-800 bg-slate-900/55 p-5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  HRV · RMSSD
                </span>
                <HeartPulse size={16} className="text-violet-300" />
              </div>
              <div className="mt-4 flex items-baseline gap-2">
                <span className="font-mono text-4xl font-semibold tabular-nums text-slate-100">
                  {formatVital(vitals.rmssdMs)}
                </span>
                <span className="text-sm text-slate-500">ms</span>
              </div>
              <div className="mt-4">
                <span
                  className={`inline-flex rounded-full border px-2.5 py-1 text-[10px] font-semibold ${stressColor(vitals.autonomicStress)}`}
                >
                  {vitals.autonomicStress ?? 'Awaiting valid intervals'}
                </span>
              </div>
            </article>

            <article className="rounded-2xl border border-slate-800 bg-slate-900/55 p-5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Respiratory rate
                </span>
                <Activity size={16} className="text-cyan-300" />
              </div>
              <div className="mt-4 flex items-baseline gap-2">
                <span className="font-mono text-4xl font-semibold tabular-nums text-slate-100">
                  {formatVital(vitals.respiratoryRateBrpm)}
                </span>
                <span className="text-sm text-slate-500">BrPM</span>
              </div>
              <div className="mt-4">
                <ReadoutBadge label="Pose + RSA" tone="cyan" />
              </div>
            </article>

            <article className="rounded-2xl border border-slate-800 bg-slate-900/55 p-5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Optical SpO₂ proxy
                </span>
                <Info size={16} className="text-amber-300" />
              </div>
              <div className="mt-4 flex items-baseline gap-2">
                <span className="font-mono text-4xl font-semibold tabular-nums text-slate-100">
                  {formatVital(vitals.spo2ProxyPercent, 1)}
                </span>
                <span className="text-sm text-slate-500">%</span>
              </div>
              <p className="mt-2 font-mono text-xs text-slate-400">
                R-ratio {formatVital(vitals.rRatio, 3)}
              </p>
              <details className="mt-3 border-t border-slate-800 pt-3">
                <summary className="cursor-pointer text-xs font-medium text-cyan-300">
                  RGB Research Proxy · Optical Calibration (A/B)
                </summary>
                <div className="mt-3 space-y-3">
                  <label className="block text-[11px] text-slate-400">
                    A calibration: {calibration.a.toFixed(1)}
                    <input
                      type="range"
                      min="90"
                      max="110"
                      step="0.1"
                      value={calibration.a}
                      onChange={(event) =>
                        updateCalibration('a', Number(event.currentTarget.value))
                      }
                      className="mt-1.5 block w-full accent-cyan-400"
                    />
                  </label>
                  <label className="block text-[11px] text-slate-400">
                    B calibration: {calibration.b.toFixed(1)}
                    <input
                      type="range"
                      min="0"
                      max="30"
                      step="0.1"
                      value={calibration.b}
                      onChange={(event) =>
                        updateCalibration('b', Number(event.currentTarget.value))
                      }
                      className="mt-1.5 block w-full accent-cyan-400"
                    />
                  </label>
                </div>
              </details>
            </article>
          </div>
        </section>

        <footer className="mt-6 flex flex-wrap items-center justify-between gap-2 border-t border-slate-900 pt-4 text-[10px] uppercase tracking-wider text-slate-600">
          <span>VitalsLens-AI · Experimental rPPG research interface</span>
          <span>Not a medical device · Do not use for clinical decisions</span>
        </footer>
      </div>
    </main>
  )
}

function makeSignalSample(frame: VisionFrame): SignalSample | null {
  const sample = frame.extraction.rgbSample
  if (!frame.extraction.isFaceValid || !sample || !sample.background) return null
  return {
    timestamp: frame.timestampMs,
    forehead: sample.forehead,
    leftCheek: sample.leftCheek,
    rightCheek: sample.rightCheek,
    background: sample.background,
    shoulderY: frame.extraction.shoulderY,
    headVelocity: frame.extraction.headVelocity,
  }
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-xs text-slate-500">
      <span className={`h-2 w-2 rounded-full ${color}`} />
      {label}
    </span>
  )
}

function ReadoutBadge({
  label,
  tone,
}: {
  label: string
  tone: 'cyan' | 'slate'
}) {
  const toneClass =
    tone === 'cyan'
      ? 'border-cyan-400/20 bg-cyan-400/5 text-cyan-300'
      : 'border-slate-700 bg-slate-800/60 text-slate-400'
  return (
    <span
      className={`inline-flex rounded-full border px-2.5 py-1 text-[10px] font-medium ${toneClass}`}
    >
      {label}
    </span>
  )
}

export default App

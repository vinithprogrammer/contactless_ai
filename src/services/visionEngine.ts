import {
  FaceLandmarker,
  FilesetResolver,
  PoseLandmarker,
} from '@mediapipe/tasks-vision'
import {
  extractRoiSample,
  type Landmark,
  type RoiExtraction,
} from '../utils/roiExtractor'

const TASKS_VISION_VERSION = '0.10.35'
const WASM_FILES_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`
const FACE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
const POSE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'

export interface VisionFrame {
  extraction: RoiExtraction
  timestampMs: number
  width: number
  height: number
}

export interface VisionEngine {
  start(
    video: HTMLVideoElement,
    processingCanvas: HTMLCanvasElement,
    onFrame: (frame: VisionFrame) => void,
    onError: (error: unknown) => void,
  ): () => void
  close(): void
}

function toLandmarks(
  landmarks: { x: number; y: number; z?: number; visibility?: number }[] | undefined,
): Landmark[] | null {
  return landmarks?.map(({ x, y, z, visibility }) => ({
    x,
    y,
    z,
    visibility,
  })) ?? null
}

export async function initializeVisionEngine(): Promise<VisionEngine> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_FILES_URL)
  const faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: FACE_MODEL_URL },
    runningMode: 'VIDEO',
    numFaces: 1,
  })
  const poseLandmarker = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: POSE_MODEL_URL },
    runningMode: 'VIDEO',
    numPoses: 1,
  })

  return {
    start(video, processingCanvas, onFrame, onError) {
      const context = processingCanvas.getContext('2d', {
        willReadFrequently: true,
      })
      if (!context) throw new Error('Unable to create the video processing canvas context.')

      let animationFrameId = 0
      let stopped = false
      let previousNose: { x: number; y: number } | null = null

      const processFrame = () => {
        if (stopped) return
        animationFrameId = requestAnimationFrame(processFrame)
        if (
          video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
          video.videoWidth === 0 ||
          video.videoHeight === 0
        ) {
          return
        }

        try {
          const width = video.videoWidth
          const height = video.videoHeight
          if (processingCanvas.width !== width || processingCanvas.height !== height) {
            processingCanvas.width = width
            processingCanvas.height = height
          }
          context.drawImage(video, 0, 0, width, height)
          const imageData = context.getImageData(0, 0, width, height)
          const timestampMs = performance.now()
          const faceResult = faceLandmarker.detectForVideo(video, timestampMs)
          const poseResult = poseLandmarker.detectForVideo(video, timestampMs)
          const faceLandmarks = toLandmarks(faceResult.faceLandmarks[0])
          const poseLandmarks = toLandmarks(poseResult.landmarks[0])
          const extraction = extractRoiSample(
            imageData,
            faceLandmarks,
            poseLandmarks,
            previousNose,
          )
          const nose = faceLandmarks?.[1]
          previousNose =
            extraction.isFaceValid && nose ? { x: nose.x, y: nose.y } : null

          onFrame({ extraction, timestampMs, width, height })
        } catch (error) {
          stopped = true
          cancelAnimationFrame(animationFrameId)
          onError(error)
        }
      }

      animationFrameId = requestAnimationFrame(processFrame)
      return () => {
        stopped = true
        cancelAnimationFrame(animationFrameId)
      }
    },
    close() {
      faceLandmarker.close()
      poseLandmarker.close()
    },
  }
}

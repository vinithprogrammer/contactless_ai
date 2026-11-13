export interface Landmark {
  x: number
  y: number
  z?: number
  visibility?: number
}

export interface RGB {
  r: number
  g: number
  b: number
}

export interface PixelRegion {
  count: number
  mean: RGB | null
}

export interface RoiExtraction {
  isFaceValid: boolean
  rgbSample: {
    forehead: RGB
    leftCheek: RGB
    rightCheek: RGB
    background: RGB | null
  } | null
  forehead: PixelRegion
  leftCheek: PixelRegion
  rightCheek: PixelRegion
  background: PixelRegion
  headVelocity: number | null
  shoulderY: number | null
  faceLandmarks: Landmark[] | null
}

export const ROI_LANDMARKS = {
  forehead: [10, 67, 109, 151, 338, 297],
  leftCheek: [116, 117, 118, 101, 50, 123],
  rightCheek: [345, 346, 347, 330, 280, 352],
} as const

export const BACKGROUND_ROI = { x: 10, y: 10, width: 40, height: 40 }

const MIN_VALID_PIXELS = 25
const GLARE_CHANNEL_THRESHOLD = 240
const MIN_LUMINANCE = 30

interface Point {
  x: number
  y: number
}

interface Rect {
  left: number
  top: number
  right: number
  bottom: number
}

function emptyRegion(): PixelRegion {
  return { count: 0, mean: null }
}

function makePolygon(
  landmarks: Landmark[],
  indices: readonly number[],
  width: number,
  height: number,
): Point[] {
  return indices.flatMap((index) => {
    const landmark = landmarks[index]
    return landmark && Number.isFinite(landmark.x) && Number.isFinite(landmark.y)
      ? [{ x: landmark.x * width, y: landmark.y * height }]
      : []
  })
}

function getFaceBounds(
  landmarks: Landmark[],
  width: number,
  height: number,
): Rect | null {
  const points = landmarks
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
    .map((point) => ({ x: point.x * width, y: point.y * height }))
  if (points.length === 0) return null

  return {
    left: Math.min(...points.map(({ x }) => x)),
    top: Math.min(...points.map(({ y }) => y)),
    right: Math.max(...points.map(({ x }) => x)),
    bottom: Math.max(...points.map(({ y }) => y)),
  }
}

function intersects(rect: Rect, bounds: Rect): boolean {
  return (
    rect.left < bounds.right &&
    rect.right > bounds.left &&
    rect.top < bounds.bottom &&
    rect.bottom > bounds.top
  )
}

function pointInPolygon(point: Point, polygon: Point[]): boolean {
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const current = polygon[i]
    const previous = polygon[j]
    const crosses =
      current.y > point.y !== previous.y > point.y &&
      point.x <
        ((previous.x - current.x) * (point.y - current.y)) /
          (previous.y - current.y) +
          current.x
    if (crosses) inside = !inside
  }
  return inside
}

function extractPolygonPixels(imageData: ImageData, polygon: Point[]): PixelRegion {
  if (polygon.length < 3) return emptyRegion()

  const { width, height, data } = imageData
  const left = Math.max(0, Math.floor(Math.min(...polygon.map(({ x }) => x))))
  const top = Math.max(0, Math.floor(Math.min(...polygon.map(({ y }) => y))))
  const right = Math.min(width, Math.ceil(Math.max(...polygon.map(({ x }) => x))))
  const bottom = Math.min(height, Math.ceil(Math.max(...polygon.map(({ y }) => y))))
  let count = 0
  let red = 0
  let green = 0
  let blue = 0

  for (let y = top; y < bottom; y += 1) {
    for (let x = left; x < right; x += 1) {
      if (!pointInPolygon({ x: x + 0.5, y: y + 0.5 }, polygon)) continue
      const offset = (y * width + x) * 4
      const r = data[offset]
      const g = data[offset + 1]
      const b = data[offset + 2]
      const luminance = 0.299 * r + 0.587 * g + 0.114 * b
      if (
        r > GLARE_CHANNEL_THRESHOLD ||
        g > GLARE_CHANNEL_THRESHOLD ||
        b > GLARE_CHANNEL_THRESHOLD ||
        luminance < MIN_LUMINANCE
      ) {
        continue
      }

      red += r
      green += g
      blue += b
      count += 1
    }
  }

  return {
    count,
    mean:
      count === 0
        ? null
        : { r: red / count, g: green / count, b: blue / count },
  }
}

function extractBackgroundPixels(
  imageData: ImageData,
  faceBounds: Rect | null,
): PixelRegion {
  const { x, y, width, height } = BACKGROUND_ROI
  const bounds = { left: x, top: y, right: x + width, bottom: y + height }
  if (faceBounds && intersects(bounds, faceBounds)) return emptyRegion()

  const clipped = {
    left: Math.max(0, bounds.left),
    top: Math.max(0, bounds.top),
    right: Math.min(imageData.width, bounds.right),
    bottom: Math.min(imageData.height, bounds.bottom),
  }
  let count = 0
  let red = 0
  let green = 0
  let blue = 0
  for (let y = clipped.top; y < clipped.bottom; y += 1) {
    for (let x = clipped.left; x < clipped.right; x += 1) {
      const offset = (y * imageData.width + x) * 4
      const r = imageData.data[offset]
      const g = imageData.data[offset + 1]
      const b = imageData.data[offset + 2]
      const luminance = 0.299 * r + 0.587 * g + 0.114 * b
      if (
        r > GLARE_CHANNEL_THRESHOLD ||
        g > GLARE_CHANNEL_THRESHOLD ||
        b > GLARE_CHANNEL_THRESHOLD ||
        luminance < MIN_LUMINANCE
      ) {
        continue
      }

      red += r
      green += g
      blue += b
      count += 1
    }
  }
  return {
    count,
    mean:
      count === 0
        ? null
        : { r: red / count, g: green / count, b: blue / count },
  }
}

function isVisibleInFrame(landmark: Landmark | undefined): landmark is Landmark {
  return Boolean(
    landmark &&
      landmark.x >= 0 &&
      landmark.x <= 1 &&
      landmark.y >= 0 &&
      landmark.y <= 1 &&
      (landmark.visibility === undefined || landmark.visibility >= 0.5),
  )
}

function getShoulderY(
  faceLandmarks: Landmark[],
  poseLandmarks: Landmark[] | null,
): number | null {
  const leftShoulder = poseLandmarks?.[11]
  const rightShoulder = poseLandmarks?.[12]
  if (isVisibleInFrame(leftShoulder) && isVisibleInFrame(rightShoulder)) {
    return (leftShoulder.y + rightShoulder.y) / 2
  }
  const chin = faceLandmarks[152]
  return isVisibleInFrame(chin) ? chin.y : null
}

export function extractRoiSample(
  imageData: ImageData,
  faceLandmarks: Landmark[] | null,
  poseLandmarks: Landmark[] | null,
  previousNose: Point | null,
): RoiExtraction {
  if (!faceLandmarks || faceLandmarks.length === 0) {
    return {
      isFaceValid: false,
      rgbSample: null,
      forehead: emptyRegion(),
      leftCheek: emptyRegion(),
      rightCheek: emptyRegion(),
      background: emptyRegion(),
      headVelocity: null,
      shoulderY: null,
      faceLandmarks: null,
    }
  }

  const { width, height } = imageData
  const foreheadPolygon = makePolygon(
    faceLandmarks,
    ROI_LANDMARKS.forehead,
    width,
    height,
  )
  const leftCheekPolygon = makePolygon(
    faceLandmarks,
    ROI_LANDMARKS.leftCheek,
    width,
    height,
  )
  const rightCheekPolygon = makePolygon(
    faceLandmarks,
    ROI_LANDMARKS.rightCheek,
    width,
    height,
  )
  const forehead = extractPolygonPixels(imageData, foreheadPolygon)
  const leftCheek = extractPolygonPixels(imageData, leftCheekPolygon)
  const rightCheek = extractPolygonPixels(imageData, rightCheekPolygon)
  const background = extractBackgroundPixels(
    imageData,
    getFaceBounds(faceLandmarks, width, height),
  )
  const isFaceValid =
    forehead.count >= MIN_VALID_PIXELS &&
    leftCheek.count >= MIN_VALID_PIXELS &&
    rightCheek.count >= MIN_VALID_PIXELS

  const nose = faceLandmarks[1]
  const leftEye = faceLandmarks[33]
  const rightEye = faceLandmarks[263]
  let headVelocity: number | null = null
  if (
    isFaceValid &&
    previousNose &&
    nose &&
    leftEye &&
    rightEye &&
    width > 0 &&
    height > 0
  ) {
    const interOcularWidth =
      Math.hypot(
        (rightEye.x - leftEye.x) * width,
        (rightEye.y - leftEye.y) * height,
      ) || 0
    if (interOcularWidth > 0) {
      headVelocity =
        Math.hypot(
          (nose.x - previousNose.x) * width,
          (nose.y - previousNose.y) * height,
        ) / interOcularWidth
    }
  }

  return {
    isFaceValid,
    rgbSample:
      isFaceValid && forehead.mean && leftCheek.mean && rightCheek.mean
        ? {
            forehead: forehead.mean,
            leftCheek: leftCheek.mean,
            rightCheek: rightCheek.mean,
            background: background.mean,
          }
        : null,
    forehead,
    leftCheek,
    rightCheek,
    background,
    headVelocity,
    shoulderY: getShoulderY(faceLandmarks, poseLandmarks),
    faceLandmarks,
  }
}

/**
 * @file skin_smoothing.cpp
 * @brief Automatic skin smoothing and blemish removal using OpenCV.
 *
 * This tool:
 *  1. Detects face and eyes with Haar cascades.
 *  2. Builds a skin‐color model from the detected face region.
 *  3. Generates a binary skin mask (HSV thresholds + morphology).
 *  4. Refines the mask with GrabCut.
 *  5. Detects blemishes via gradient + blob detection within the skin region.
 *  6. Removes blemishes using seamlessClone with low‐texture patches.
 *  7. Applies bilateral smoothing to the corrected face area.
 */

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <vector>

#include "config.pch"

//--------------------------------------------------------------------------------------
// Configuration
//--------------------------------------------------------------------------------------
static const std::filesystem::path kDataDir = DATA_DIR;
static const std::string kFaceModel = "models/haarcascade_frontalface_default.xml";
static const std::string kEyeModel = "models/haarcascade_eye.xml";

static const cv::Size kMinFaceSize = {100, 100};
static const cv::Size kMinEyeSize = {120, 120};
static constexpr double kSigmaFactor = 2.5;
static constexpr int kHistBinsH = 30;
static constexpr int kHistBinsS = 32;
static constexpr int kMorphKernel = 3;
static constexpr int kMorphIter = 2;
static constexpr int kGCIter = 2;
static constexpr float kMinBlemPct = 0.01f;
static constexpr float kMaxBlemPct = 0.06f;

//--------------------------------------------------------------------------------------
// Utility functions
//--------------------------------------------------------------------------------------

/**
 * @brief Loads an image from the specified filesystem path.
 *        Exits the program with an error message if loading fails.
 *
 * @param path  Filesystem path to the image file.
 * @return      A cv::Mat containing the loaded image.
 */
static cv::Mat loadOrExit(const std::filesystem::path& path) {
  cv::Mat img = cv::imread(path.string());
  if (img.empty()) {
    std::cerr << "ERROR: Cannot load " << path << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return img;
}

/**
 * @brief Displays an image in a resizable window and waits for a key press.
 *
 * Opens a window titled `win`, resizes it to 800×600, shows `img`,
 * then blocks until any key is pressed before destroying the window.
 *
 * @param win  Title of the display window.
 * @param img  Image to display.
 */
static void show(const std::string& win, const cv::Mat& img) {
  cv::namedWindow(win, cv::WINDOW_NORMAL);
  cv::resizeWindow(win, 800, 600);
  cv::imshow(win, img);
  cv::waitKey(0);
  cv::destroyWindow(win);
}

/**
 * @brief Computes the texture variance of a patch by applying a Laplacian
 *        filter to the hue channel and summing the squared responses.
 *
 * @param patch  Input BGR image patch.
 * @return       Sum of squared Laplacian values as a measure of variance.
 */
static double patchVariance(const cv::Mat& patch) {
  cv::Mat hsv, lap, sq;
  cv::cvtColor(patch, hsv, cv::COLOR_BGR2HSV);
  cv::Mat h;
  cv::extractChannel(hsv, h, 0);
  cv::Laplacian(h, lap, CV_32F, 3);
  cv::pow(lap, 2, sq);
  return cv::sum(sq)[0];
}

/**
 * @brief Searches eight directions around a center point to find the
 *        patch of size (2*radius)×(2*radius) with the lowest texture variance.
 *
 * @param img     Source image from which patches are extracted.
 * @param center  Center point around which to search for replacement patches.
 * @param radius  Radius of the square patch (half the side length).
 * @return        The best‐matching patch (cloned); returns an empty Mat if none found.
 */
static cv::Mat bestLowTexturePatch(const cv::Mat& img, cv::Point center, int radius) {
  cv::Mat best;
  double bestVar = std::numeric_limits<double>::infinity();
  int diameter = 2 * radius;

  static const std::vector<cv::Point2f> dirs = {{1, 0},
                                                {-1, 0},
                                                {0, 1},
                                                {0, -1},
                                                {0.707f, 0.707f},
                                                {-0.707f, 0.707f},
                                                {0.707f, -0.707f},
                                                {-0.707f, -0.707f}};

  for (auto d : dirs) {
    cv::Point c = center + cv::Point(cvRound(d.x * diameter), cvRound(d.y * diameter));
    int x0 = c.x - radius, y0 = c.y - radius;
    if (x0 < 0 || y0 < 0 || x0 + diameter > img.cols || y0 + diameter > img.rows) continue;
    cv::Mat patch = img(cv::Rect{x0, y0, diameter, diameter});
    double v = patchVariance(patch);
    if (v < bestVar) {
      bestVar = v;
      best = patch.clone();
    }
  }
  return best;
}

/**
 * @brief Removes blemishes by seamless‐cloning low‐texture patches over each keypoint.
 *
 * For each keypoint in `kps`, a patch of radius ~kp.size is found via
 * bestLowTexturePatch() and seamlessly cloned back into `src` at the keypoint location.
 *
 * @param src   Source image that will be modified in‐place.
 * @param kps   Vector of cv::KeyPoint indicating blemish locations and sizes.
 */
static void removeBlemishes(cv::Mat& src, const std::vector<cv::KeyPoint>& kps) {
  for (auto& kp : kps) {
    cv::Point c{cvRound(kp.pt.x), cvRound(kp.pt.y)};
    int r = cvRound(kp.size) * 1.25;
    cv::Mat patch = bestLowTexturePatch(src, c, r);
    if (patch.empty()) continue;
    cv::Mat mask = cv::Mat::zeros(patch.size(), CV_8U);
    cv::circle(mask, {r, r}, r, 255, cv::FILLED);
    cv::seamlessClone(patch, src, mask, c, src, cv::NORMAL_CLONE_WIDE);
    // cv::circle(src, c, r, 255, cv::FILLED);
  }
}

/**
 * @brief Generates a binary face mask that includes the detected face region
 *        (as a rectangle down to the bottom of the image) and excludes eyes and nose
 *        using filled ellipses.
 *
 * @param imageSize  Size of the source image.
 * @param faceRect   Bounding rectangle of the detected face.
 * @param eyeRects   Vector of bounding rectangles for each detected eye.
 * @return           Single-channel 8-bit mask:
 *                    - 255 inside the face rectangle,
 *                    - 0 in the eye and nose ellipses,
 *                    - 0 elsewhere.
 */
cv::Mat createFaceMask(const cv::Size& imageSize, const cv::Rect& faceRect,
                       const std::vector<cv::Rect>& eyeRects) {
  // Start with a blank mask
  cv::Mat mask(imageSize, CV_8U, cv::Scalar(0));

  // 1) Draw face region as a filled rectangle from top of face to bottom of image
  cv::rectangle(mask, cv::Point(faceRect.x, faceRect.y),
                cv::Point(faceRect.x + faceRect.width, imageSize.height), cv::Scalar(255),
                cv::FILLED);

  // 2) Exclude each eye area with an enlarged filled ellipse
  std::vector<cv::Point> eyeCenters;
  for (const auto& e : eyeRects) {
    cv::Point center(e.x + e.width / 2, e.y + e.height / 3);
    eyeCenters.push_back(center);
    cv::Size axes(static_cast<int>(1.25 * e.width / 2.0), e.height / 2);
    cv::ellipse(mask, center, axes, 0.0, 0.0, 360.0, cv::Scalar(0), cv::FILLED);
  }

  // 3) If two or more eyes, sort by x to identify left/right, then carve out nose ellipse
  if (eyeCenters.size() >= 2) {
    // Only consider the two most prominent eyes:
    cv::Point c1 = eyeCenters[0], c2 = eyeCenters[1];
    // Determine which is left/right by x-coordinate
    cv::Point leftEyeCenter = (c1.x < c2.x) ? c1 : c2;
    cv::Point rightEyeCenter = (c1.x < c2.x) ? c2 : c1;

    // Nose center midway horizontally, slightly below eyes
    cv::Point noseCenter(
        (leftEyeCenter.x + rightEyeCenter.x) / 2,
        (leftEyeCenter.y + rightEyeCenter.y) / 2 + static_cast<int>(faceRect.height * 0.3));
    int noseWidth = std::abs(rightEyeCenter.x - leftEyeCenter.x);
    int noseHeight = static_cast<int>(faceRect.height * 0.15);
    cv::Size noseAxes(static_cast<int>(noseWidth / 4.5), noseHeight / 2);

    cv::ellipse(mask, noseCenter, noseAxes, 0.0, 0.0, 360.0, cv::Scalar(0), cv::FILLED);
  }

  return mask;
}

/**
 * @brief Processes a single image: detects face/eyes, builds skin mask,
 *        removes blemishes, applies smoothing, and returns both the
 *        intermediate (after blemish removal) and final images.
 *
 * @param image            Input BGR image.
 * @param faceC            Loaded face CascadeClassifier.
 * @param eyeC             Loaded eye CascadeClassifier.
 * @param outRemoved       Output: image after removeBlemishes().
 * @param outFinal         Output: final smoothed image.
 */
void processSkinSmoothing(const cv::Mat& image, cv::CascadeClassifier& faceC,
                          cv::CascadeClassifier& eyeC, cv::Mat& outRemoved, cv::Mat& outFinal) {
  // 1. Face & eye detection
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  std::vector<cv::Rect> faces, eyes;
  faceC.detectMultiScale(gray, faces, 1.1, 3, 0, kMinFaceSize);
  eyeC.detectMultiScale(gray, eyes, 1.1, 3, 0, kMinEyeSize);
  if (faces.empty()) throw std::runtime_error("No face detected");
  cv::Rect face = faces.front();
  eyeC.detectMultiScale(gray, eyes, 1.1, 3, 0, kMinEyeSize);

  // Build face mask (exclude eyes and nose)
  cv::Mat faceMask = createFaceMask(image.size(), face, eyes);

  // Smooth for color sampling
  cv::Mat blurImage;
  cv::GaussianBlur(image, blurImage, {17, 17}, 6);

  // Sample central region of face
  int y0 = face.y + cvRound(face.height * 0.10f);
  int y1 = face.y + cvRound(face.height * 0.80f);
  int x0 = face.x + cvRound(face.width * 0.20f);
  int x1 = face.x + cvRound(face.width * 0.80f);
  cv::Mat skinSample = blurImage(cv::Rect{x0, y0, x1 - x0, y1 - y0});

  // Build 2D HSV histogram
  cv::Mat hsv;
  cv::cvtColor(skinSample, hsv, cv::COLOR_BGR2HSV);
  int histSizes[] = {kHistBinsH, kHistBinsS};
  float hRanges[] = {0, 180}, sRanges[] = {0, 256};
  const float* ranges[] = {hRanges, sRanges};
  int channels[] = {0, 1};
  cv::Mat hist;
  cv::calcHist(&hsv, 1, channels, {}, hist, 2, histSizes, ranges);

  // Reduce to 1D histograms
  cv::Mat histH, histS;
  cv::reduce(hist, histH, 1, cv::REDUCE_SUM);
  cv::reduce(hist, histS, 0, cv::REDUCE_SUM);
  double total = cv::sum(hist)[0];

  auto computeStats = [&](const cv::Mat& h, int bins, double minV, double maxV) {
    double mean = 0, var = 0, binW = (maxV - minV) / bins;
    for (int i = 0; i < bins; ++i) {
      double p = h.at<float>(i) / total;
      double c = minV + (i + 0.5) * binW;
      mean += c * p;
      var += c * c * p;
    }
    var -= mean * mean;
    return std::make_pair(mean, std::sqrt(var));
  };

  auto [meanH, stdH] = computeStats(histH, kHistBinsH, 0, 180);
  auto [meanS, stdS] = computeStats(histS, kHistBinsS, 0, 256);

  // Correctly use cv::Scalar for inRange
  float hLo = static_cast<float>(std::max(0.0, meanH - kSigmaFactor * stdH));
  float hHi = static_cast<float>(std::min(180.0, meanH + kSigmaFactor * stdH));
  float sLo = static_cast<float>(std::max(0.0, meanS - kSigmaFactor * stdS));
  float sHi = static_cast<float>(std::min(256.0, meanS + kSigmaFactor * stdS));

  cv::cvtColor(blurImage, hsv, cv::COLOR_BGR2HSV);
  cv::Mat skinMask;
  cv::inRange(hsv, cv::Scalar(hLo, sLo, 50), cv::Scalar(hHi, sHi, 255), skinMask);
  cv::bitwise_and(skinMask, faceMask, skinMask);

  // Morphology + GrabCut refinement
  cv::Mat morph = skinMask;
  cv::morphologyEx(skinMask, morph, cv::MORPH_OPEN,
                   cv::getStructuringElement(cv::MORPH_RECT, {kMorphKernel, kMorphKernel}), {},
                   kMorphIter);

  cv::Mat gcMask(image.size(), CV_8U, cv::GC_PR_BGD);
  gcMask.setTo(cv::GC_PR_FGD, morph);
  cv::grabCut(image, gcMask, {}, cv::Mat(), cv::Mat(), kGCIter, cv::GC_INIT_WITH_MASK);
  cv::Mat refinedMask = (gcMask == cv::GC_FGD) | (gcMask == cv::GC_PR_FGD);

  // Remove unwanted borders from the image
  cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE,
                                             cv::Size(kMorphKernel * 2, kMorphKernel));
  cv::morphologyEx(refinedMask, refinedMask, cv::MorphTypes::MORPH_DILATE, kernel,
                   cv::Point(-1, -1), kMorphIter);

  // Blemish detection via gradient + blobs
  cv::Mat hsv2, grayV;
  cv::cvtColor(image, hsv2, cv::COLOR_BGR2HSV);
  cv::extractChannel(hsv2, grayV, 2);
  int w = face.width;
  int minB = cvRound(w * kMinBlemPct), maxB = cvRound(w * kMaxBlemPct);
  minB += (minB % 2 == 0);
  maxB += (maxB % 2 == 0);

  kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(minB * 2, minB));
  cv::morphologyEx(refinedMask, refinedMask, cv::MorphTypes::MORPH_ERODE, kernel, cv::Point(-1, -1),
                   kMorphIter * 2);
  cv::bitwise_and(refinedMask, morph, refinedMask);

  cv::Mat gx, gy, mag;
  cv::Sobel(grayV, gx, CV_32F, 1, 0, minB);
  cv::Sobel(grayV, gy, CV_32F, 0, 1, minB);
  cv::normalize(gx, gx, 1.0f, -1.0f, cv::NormTypes::NORM_MINMAX);
  cv::normalize(gy, gy, 1.0f, -1.0f, cv::NormTypes::NORM_MINMAX);
  cv::magnitude(gx, gy, mag);
  cv::normalize(mag, mag, 2.0, -2.0, cv::NORM_MINMAX);
  mag = cv::abs(mag);
  mag = mag - 1;

  cv::multiply(mag, refinedMask, mag, (1.0f), CV_32F);
  cv::Mat mag8;
  mag.convertTo(mag8, CV_8U, 1.0);
  cv::GaussianBlur(mag8, mag8, {minB, minB}, 3);
  // show("Que necio!!", mag8);

  mag8.setTo(0, ~refinedMask);

  cv::SimpleBlobDetector::Params bp;
  bp.filterByArea = true;
  bp.minArea = minB * minB;
  bp.maxArea = maxB * maxB;
  bp.filterByCircularity = true;
  bp.minCircularity = 0.35f;
  bp.filterByInertia = true;
  bp.minInertiaRatio = 0.15f;
  bp.filterByConvexity = false;
  bp.filterByColor = false;
  bp.thresholdStep = 5;
  bp.minThreshold = 0;
  bp.maxThreshold = 200;

  auto blobDet = cv::SimpleBlobDetector::create(bp);
  std::vector<cv::KeyPoint> keypoints;
  blobDet->detect(mag8, keypoints);
  cv::Mat annotated_keypoints;
  cv::drawKeypoints(image, keypoints, annotated_keypoints, (255, 0, 0),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::Mat working = image.clone();
  removeBlemishes(working, keypoints);

  // Save intermediate
  outRemoved = working.clone();

  // 5. Final smoothing & composite
  cv::Mat smooth;
  cv::bilateralFilter(working, smooth, -1, 35.0, 20.0);
  cv::Mat result = image.clone();
  cv::seamlessClone(smooth, working, refinedMask, cv::Point(smooth.cols / 2, smooth.rows / 2),
                    result, cv::SeamlessCloneFlags::NORMAL_CLONE_WIDE);

  outFinal = result.clone();
}

//--------------------------------------------------------------------------------------
// Main pipeline
//--------------------------------------------------------------------------------------
int main() {
  // Load models once
  cv::CascadeClassifier faceC, eyeC;
  faceC.load((kDataDir / "../data" / kFaceModel).string());
  eyeC.load((kDataDir / "../data" / kEyeModel).string());

  std::vector<std::string> imageNames = {"img4.png", "img2.png", "img3.png", "img1.png"};
  for (const auto& name : imageNames) {
    try {
      auto inPath = kDataDir / "../data" / name;
      auto removedName = "removed_" + name;
      auto finalName = "smoothed_" + name;

      cv::Mat image = loadOrExit(inPath);
      cv::Mat removed, finalImg;
      processSkinSmoothing(image, faceC, eyeC, removed, finalImg);

      // Save
      cv::imwrite((kDataDir / "../data" / removedName).string(), removed);
      cv::imwrite((kDataDir / "../data" / finalName).string(), finalImg);

      // Show
      std::vector<cv::Mat> panels = {image, removed, finalImg};
      cv::Mat combined;
      cv::hconcat(panels, combined);
      show("Original | Removed | Smoothed — " + name, combined);
    } catch (const std::exception& e) {
      std::cerr << "Error on " << name << ": " << e.what() << "\n";
    }
  }
  return EXIT_SUCCESS;
}
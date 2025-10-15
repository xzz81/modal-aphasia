import abc

import cv2
import numpy as np

from modal_aphasia.data import constants as _constants


class ConceptClassifier(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def classify(self, image: np.ndarray) -> str | None:
        pass


class ShapeClassifier(ConceptClassifier):
    """
    Classifier for detecting shapes in generated images.
    Based on the shapes defined in constants.py
    """

    def __init__(self):
        self.shapes = tuple(_constants.CONCEPT_TO_SYNTHETIC_MAP["shape"].keys())

    def classify(self, image: np.ndarray) -> str | None:
        """
        Main function to classify the shape in an image.

        Args:
            image: numpy array (RGB format)
        """

        # Convert RGB to BGR (OpenCV uses BGR)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # For black filled shapes, use thresholding instead of edge detection
        # Black shapes will have low pixel values
        _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)

        # Clean up the binary image (remove small noise)
        kernel = np.ones((3, 3), np.uint8)
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel)

        # Find connected components and keep only the largest one
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_clean, connectivity=8)

        if num_labels <= 1:
            return None

        # Find the largest component (excluding background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_component = (labels == largest_label).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(largest_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour with higher precision for filled shapes
        epsilon = 0.03 * cv2.arcLength(largest_contour, True)  # More precise approximation
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Classify the shape using the same logic as detect_shape.py
        shape_name = self._classify_contour(largest_contour, approx)

        assert shape_name is None or shape_name in self.shapes, "Shape name is not in the list of shapes"
        return shape_name

    def _classify_contour(self, contour, approx) -> str | None:
        """
        Classify a contour into a specific shape.
        """

        # Try each shape detection method in order of complexity
        if self._is_plus(contour):
            return "plus"
        elif self._is_star(contour):
            return "star"
        elif self._is_circle(contour, approx):
            return "circle"
        elif len(approx) == 3:
            return "triangle"
        elif len(approx) == 4:
            return "square"
        elif len(approx) == 5:
            return "pentagon"
        elif len(approx) == 6:
            return "hexagon"

        # If no specific shape is detected
        return None

    def _is_circle(self, contour, approx) -> bool:
        """
        Circle detection using fill ratio.
        """
        area = cv2.contourArea(contour)
        if area == 0:
            return False

        # Get minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * (radius**2)
        fill_ratio = area / circle_area

        # Circle should have high fill ratio and many vertices
        return fill_ratio > 0.75 and len(approx) > 6

    def _is_plus(self, contour) -> bool:
        """
        Plus detection using aspect ratio and fill ratio.
        """
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        # Plus should be roughly square
        if not (0.85 <= aspect_ratio <= 1.15):
            return False

        rect_area = w * h
        area = cv2.contourArea(contour)
        fill_ratio = area / rect_area

        # Plus should have specific fill ratio
        if not (0.30 <= fill_ratio <= 0.75):
            return False

        # Approximate contour to vertices
        vertices = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        pts = np.squeeze(vertices)

        if len(pts) < 8:  # Plus shape has multiple vertices
            return False

        # Function to get edge angle
        def edge_angle(p1, p2):
            dx, dy = p2 - p1
            return np.degrees(np.arctan2(dy, dx)) % 180

        # Count orthogonal edges
        orthogonal_edges = 0
        edge_angles = []
        for i in range(len(pts)):
            angle = edge_angle(pts[i], pts[(i + 1) % len(pts)])
            edge_angles.append(angle)
            if any(abs(angle - t) < 10 for t in (0, 90)):
                orthogonal_edges += 1

        orthogonal_ratio = orthogonal_edges / len(pts)

        # Require majority of edges to be orthogonal
        if orthogonal_ratio < 0.8:
            return False

        return True

    def _is_star(self, contour) -> bool:
        """
        Star detection using convexity and peak analysis.
        """
        if cv2.isContourConvex(contour):
            return False

        # Approximate contour
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) < 8:
            return False

        # Find centroid
        M = cv2.moments(approx)
        if M["m00"] == 0:
            return False
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = np.array([cx, cy])

        # Sort vertices by polar angle around centroid
        pts = approx.reshape(-1, 2)
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        pts = pts[np.argsort(angles)]

        # Distances from centroid
        distances = np.linalg.norm(pts - center, axis=1)

        # Detect peaks (outer tips) and valleys (inner points)
        peaks = []
        valleys = []
        for i in range(len(distances)):
            prev = distances[i - 1]
            curr = distances[i]
            nxt = distances[(i + 1) % len(distances)]
            if curr > prev and curr > nxt:
                peaks.append(i)
            elif curr < prev and curr < nxt:
                valleys.append(i)

        if len(peaks) < 5:
            return False
        if len(valleys) < 5:
            return False

        # Check ratio of peak to valley distance
        ratio = np.max(distances) / np.min(distances)
        if ratio < 1.5:
            return False

        # Sharp tip check at peaks
        acute_count = 0
        for i in peaks:
            p0 = pts[i - 1]
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            v1 = p0 - p1
            v2 = p2 - p1
            ang = abs(np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))))
            if ang < 115:
                acute_count += 1

        if acute_count < 4:
            return False

        # Also reject if average peak angle is too blunt
        if np.mean(angles) > 120:
            return False

        return True


class PatternClassifier(ConceptClassifier):
    """
    Classifier for detecting filling patterns in generated images.
    """

    def __init__(self):
        self.patterns = tuple(_constants.CONCEPT_TO_SYNTHETIC_MAP["pattern"].keys())

    def classify(self, image: np.ndarray) -> str | None:
        """
        Main function to classify the filling pattern in an image.

        Args:
            image: numpy array (RGB format)
        """

        # Convert RGB to BGR (OpenCV uses BGR)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Find the quadrant with the least black pixels (should contain the pattern)
        h, w = gray.shape
        mid_h, mid_w = h // 2, w // 2

        # Define the 4 quadrants
        quadrants = [
            gray[0:mid_h, 0:mid_w],  # Top-left
            gray[0:mid_h, mid_w:w],  # Top-right
            gray[mid_h:h, 0:mid_w],  # Bottom-left
            gray[mid_h:h, mid_w:w],  # Bottom-right
        ]

        quadrant_names = tuple(_constants.CONCEPT_TO_SYNTHETIC_MAP["position"].keys())

        # Count black pixels and calculate variance for each quadrant
        quadrant_data = []
        for i, quad in enumerate(quadrants):
            black_pixels = np.sum(quad < 15)  # Count pixels darker than 15
            variance = np.var(quad)  # Calculate variance
            quadrant_data.append(
                {
                    "index": i,
                    "name": quadrant_names[i],
                    "quadrant": quad,
                    "black_pixels": black_pixels,
                    "variance": variance,
                }
            )

        # Sort by black pixels (ascending) and get top 3 with least black pixels
        sorted_by_black = sorted(quadrant_data, key=lambda x: x["black_pixels"])
        top_3_least_black = sorted_by_black[:3]

        # Among top 3, select the one with highest variance
        best_quadrant_data = max(top_3_least_black, key=lambda x: x["variance"])

        pattern_quadrant = best_quadrant_data["quadrant"]

        # Collect all pattern detections with confidence scores
        pattern_scores = []

        # Check solid pattern (using selected quadrant)
        is_solid, solid_confidence = self._is_solid_pattern(pattern_quadrant)
        if is_solid:
            pattern_scores.append(("solid", solid_confidence))

        # Check other patterns (using selected quadrant)
        circles_pattern, circles_confidence = self._detect_circles_pattern(pattern_quadrant)
        if circles_pattern:
            pattern_scores.append(("circles", circles_confidence))

        striped_pattern, striped_confidence = self._detect_striped_pattern(pattern_quadrant)
        if striped_pattern:
            pattern_scores.append(("striped", striped_confidence))

        zigzag_pattern, zigzag_confidence = self._detect_zigzag_pattern(pattern_quadrant)
        if zigzag_pattern:
            pattern_scores.append(("zigzag", zigzag_confidence))

        checkered_pattern, checkered_confidence = self._detect_checkered_pattern(pattern_quadrant)
        if checkered_pattern:
            pattern_scores.append(("checkered", checkered_confidence))

        # Select pattern with highest confidence
        if pattern_scores:
            best_pattern, best_confidence = max(pattern_scores, key=lambda x: x[1])
            if best_confidence > 0.4:
                assert best_pattern in self.patterns, "Best pattern is not in the list of patterns"
                return best_pattern
            else:
                return None  # No pattern detected with sufficient confidence
        else:
            return None  # No specific pattern detected

    def _is_solid_pattern(self, gray: np.ndarray) -> tuple[bool, float]:
        """
        Detect solid pattern by checking variance and histogram uniformity
        Returns: (is_solid, confidence_score)
        """
        h, w = gray.shape

        # FIXME: most non-white pixels could be changed to high variance of the pixels.
        # Step 1: Find the shape position by splitting image into 4 quadrants
        # and finding which quadrant has the most non-white pixels
        mid_h, mid_w = h // 2, w // 2

        # Define the 4 quadrants
        quadrants = [
            gray[0:mid_h, 0:mid_w],  # Top-left
            gray[0:mid_h, mid_w:w],  # Top-right
            gray[mid_h:h, 0:mid_w],  # Bottom-left
            gray[mid_h:h, mid_w:w],  # Bottom-right
        ]

        # Count non-white pixels in each quadrant
        quadrant_scores = []
        for i, quad in enumerate(quadrants):
            non_white_pixels = np.sum(quad < 240)
            quadrant_scores.append((i, non_white_pixels))

        # Find the quadrant with most non-white pixels
        best_quadrant_idx, best_quadrant_score = max(quadrant_scores, key=lambda x: x[1])

        # Step 2: Determine the center point for analysis based on best quadrant
        if best_quadrant_idx == 0:  # Top-left
            center_x, center_y = mid_w // 2, mid_h // 2
        elif best_quadrant_idx == 1:  # Top-right
            center_x, center_y = mid_w + mid_w // 2, mid_h // 2
        elif best_quadrant_idx == 2:  # Bottom-left
            center_x, center_y = mid_w // 2, mid_h + mid_h // 2
        else:  # Bottom-right
            center_x, center_y = mid_w + mid_w // 2, mid_h + mid_h // 2

        # Step 3: Try different window sizes in the best region
        best_window_size = 50  # Start with minimum size
        max_ratio = 0
        best_region = None

        # Try three different window sizes
        window_sizes = [50, min(w, h) // 2, min(w, h)]

        for window_size in window_sizes:
            # Extract region around the center point
            y1 = max(0, center_y - window_size // 2)
            y2 = min(h, center_y + window_size // 2)
            x1 = max(0, center_x - window_size // 2)
            x2 = min(w, center_x + window_size // 2)

            region = gray[y1:y2, x1:x2]

            if region.size == 0:
                continue

            # Count non-white pixels and calculate ratio
            non_white_pixels = np.sum(region < 240)
            ratio = non_white_pixels / region.size

            # Update if this window has higher ratio of colored content
            if ratio > max_ratio:
                max_ratio = ratio
                best_window_size = window_size
                best_region = region.copy()

        # If no good region found, use center with default size
        if best_region is None:
            best_window_size = min(150, w // 3, h // 3)
            center_x, center_y = w // 2, h // 2
            y1 = max(0, center_y - best_window_size // 2)
            y2 = min(h, center_y + best_window_size // 2)
            x1 = max(0, center_x - best_window_size // 2)
            x2 = min(w, center_x + best_window_size // 2)
            best_region = gray[y1:y2, x1:x2]

        # Step 4: Analyze the best region
        # Compute variance
        variance = np.var(best_region)
        variance_score = max(0, 1.0 - variance / 50.0)  # Higher score for lower variance

        # Histogram check - solid should have most pixels in a narrow range
        hist = cv2.calcHist([best_region], [0], None, [256], [0, 256])
        hist = hist.flatten()
        dominant_count = np.max(hist)
        total_pixels = best_region.size
        histogram_score = dominant_count / total_pixels  # Higher score for more uniform histogram

        # Combined confidence score
        confidence = (variance_score + histogram_score) / 2.0

        return variance < 5 or histogram_score > 0.9, confidence

    def _get_enhanced_edges(self, gray):
        """
        Get enhanced edges with gap filling and dilation for better pattern detection
        """
        # Edge detection with gap filling. This line only is old version of the function.
        edges = cv2.Canny(gray, 30, 100)
        # Dilate edges to fill gaps and make them more prominent
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        # Close small gaps
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return edges

    def _detect_circles_pattern(self, gray: np.ndarray) -> tuple[bool, float]:
        """
        Detect circles pattern by looking for many small circles
        Returns: (pattern_name, confidence_score) or (None, 0.0)
        """
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=15, minRadius=8, maxRadius=20
        )

        if circles is not None:
            num_circles = len(circles[0])

            # Confidence based on number of circles detected
            if num_circles > 5:
                confidence = min(1.0, num_circles / 8.0)  # Normalize to 0-1
                return True, confidence

        return False, 0.0

    def _detect_striped_pattern(self, gray: np.ndarray) -> tuple[bool, float]:
        """
        Detect striped pattern by looking for many horizontal lines
        Returns: (pattern_name, confidence_score) or (None, 0.0)
        """

        # Edge detection
        edges = self._get_enhanced_edges(gray)

        # Probabilistic Hough Transform (returns endpoints)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=15,
            minLineLength=10,
            maxLineGap=3
        )

        horizontal_lines_list = []
        vertical_lines_list = []

        if lines is not None and len(lines) > 0:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)  # radians
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                # Normalize angle to positive range
                if angle < 0:
                    angle += np.pi

                # Horizontal check (near 0 or π) - lines parallel to x-axis
                if abs(angle) < np.pi/12 or abs(angle - np.pi) < np.pi/12:
                    angle_error = min(abs(angle), abs(angle - np.pi))
                    angle_quality = 1.0 - (angle_error / (np.pi/12))
                    horizontal_lines_list.append((x1, y1, x2, y2, length, angle_quality))

                # Vertical check (near π/2) - lines parallel to y-axis
                elif abs(angle - np.pi/2) < np.pi/12:
                    angle_error = abs(angle - np.pi/2)
                    angle_quality = 1.0 - (angle_error / (np.pi/12))
                    vertical_lines_list.append((x1, y1, x2, y2, length, angle_quality))

        horizontal_lines = len(horizontal_lines_list)
        vertical_lines = len(vertical_lines_list)

        # Confidence calculation based on horizontal vs vertical ratio
        horizontal_ratio = horizontal_lines / vertical_lines if vertical_lines > 0 else 2.5

        if horizontal_ratio > 1.5:
            confidence = 0.8 * min(1.0, horizontal_lines / 8.0) + 0.2 * min(1, horizontal_ratio - 1.5)
            return True, confidence

        return False, 0.0

    def _detect_checkered_pattern(self, gray: np.ndarray) -> tuple[bool, float]:
        """
        Detect checkered pattern by looking for regular square patterns
        Returns: (pattern_name, confidence_score) or (None, 0.0)
        """

        # Detect edges
        edges = self._get_enhanced_edges(gray)

        # Counters
        horizontal_lines = 0
        vertical_lines = 0
        horizontal_angle_quality = 0.0
        vertical_angle_quality = 0.0
        horizontal_lengths = []
        vertical_lengths = []

        # Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=15,  # fewer votes needed
            minLineLength=10,  # detect short grid edges
            maxLineGap=3,
        )

        if lines is not None and len(lines) > 0:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)  # radians
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # Normalize angle to positive range
                if angle < 0:
                    angle += np.pi

                # Vertical check (near 0 or π)
                if angle < np.pi / 12 or abs(angle - np.pi) < np.pi / 12:
                    vertical_lines += 1
                    vertical_lengths.append(length)
                    angle_error = min(angle, abs(angle - np.pi))
                    vertical_angle_quality += 1.0 - (angle_error / (np.pi / 12))

                # Horizontal check (near π/2)
                elif abs(angle - np.pi / 2) < np.pi / 12:
                    horizontal_lines += 1
                    horizontal_lengths.append(length)
                    angle_error = abs(angle - np.pi / 2)
                    horizontal_angle_quality += 1.0 - (angle_error / (np.pi / 12))

        # Quality scores
        avg_horizontal_quality = horizontal_angle_quality / horizontal_lines if horizontal_lines > 0 else 0.0
        avg_vertical_quality = vertical_angle_quality / vertical_lines if vertical_lines > 0 else 0.0

        # Check for similar numbers of horizontal and vertical lines
        line_count_ratio = (
            min(horizontal_lines, vertical_lines) / max(horizontal_lines, vertical_lines)
            if max(horizontal_lines, vertical_lines) > 0
            else 0.0
        )

        # Check for similar line lengths
        length_similarity = 0.0
        if horizontal_lengths and vertical_lengths:
            avg_horizontal_length = np.mean(horizontal_lengths)
            avg_vertical_length = np.mean(vertical_lengths)
            length_ratio = min(avg_horizontal_length, avg_vertical_length) / max(
                avg_horizontal_length, avg_vertical_length
            )
            length_similarity = length_ratio

        # Pattern decision with enhanced criteria
        min_lines = min(horizontal_lines, vertical_lines)
        max_lines = max(horizontal_lines, vertical_lines)

        if min_lines > 8 and max_lines > 8:
            line_count_score = min(1.0, min_lines / 15.0)
            angle_quality_score = (avg_horizontal_quality + avg_vertical_quality) / 2.0
            line_count_balance_score = line_count_ratio  # Higher score for balanced counts
            length_balance_score = length_similarity  # Higher score for similar lengths

            # Weighted combination: 30% line count, 20% angle quality, 25% line count balance, 25% length balance
            confidence = (
                0.3 * line_count_score
                + 0.2 * angle_quality_score
                + 0.25 * line_count_balance_score
                + 0.25 * length_balance_score
            )

            return True, confidence

        return False, 0.0

    def _detect_zigzag_pattern(self, gray: np.ndarray) -> tuple[bool, float]:
        """
        Detect zigzag pattern by looking for diagonal patterns using Probabilistic Hough Transform
        Returns: (pattern_name, confidence_score) or (None, 0.0)
        """

        # Apply edge detection (more sensitive)
        edges = self._get_enhanced_edges(gray)

        # Use probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=10,  # fewer votes needed for shorter lines
            minLineLength=8,  # detect very short diagonal segments
            maxLineGap=5,  # allow larger gaps between segments
        )

        if lines is not None and len(lines) > 0:
            # First pass: collect diagonal lines and filter close duplicates
            diagonal_lines = []
            diagonal_lengths = []  # Added for length similarity

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)
                if angle < 0:
                    angle += np.pi  # normalize to 0–π

                # Detect diagonal lines (exclude near-horizontal and near-vertical)
                if np.pi / 6 < angle < 5 * np.pi / 6 and abs(angle - np.pi / 2) > 0.3:
                    # Check if this line is too close to existing lines with similar orientation
                    too_close = False
                    for existing_line in diagonal_lines:
                        ex1, ey1, ex2, ey2 = existing_line
                        existing_angle = np.arctan2(ey2 - ey1, ex2 - ex1)
                        if existing_angle < 0:
                            existing_angle += np.pi

                        # If angles are similar (within 15 degrees)
                        if abs(angle - existing_angle) < np.pi / 12:  # ~15 degrees
                            # Calculate distance between line endpoints
                            dist1 = np.hypot(x1 - ex1, y1 - ey1)
                            dist2 = np.hypot(x1 - ex2, y1 - ey2)
                            dist3 = np.hypot(x2 - ex1, y2 - ey1)
                            dist4 = np.hypot(x2 - ex2, y2 - ey2)

                            # If any endpoint is very close to another line's endpoint (less than 20 pixels)
                            if min(dist1, dist2, dist3, dist4) < 20:
                                too_close = True
                                break

                    if not too_close:
                        diagonal_lines.append((x1, y1, x2, y2))
                        # Calculate and store line length
                        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        diagonal_lengths.append(length)

            # Second pass: process filtered diagonal lines
            diagonal_count = len(diagonal_lines)
            total_lines = len(lines)
            diagonal_angle_quality = 0.0

            for x1, y1, x2, y2 in diagonal_lines:
                angle = np.arctan2(y2 - y1, x2 - x1)
                if angle < 0:
                    angle += np.pi

                # Perfect diagonals are π/4 or 3π/4
                angle_error_45 = abs(angle - np.pi / 4)
                angle_error_135 = abs(angle - 3 * np.pi / 4)
                min_angle_error = min(angle_error_45, angle_error_135)

                diagonal_angle_quality += 1.0 - (min_angle_error / (np.pi / 6))

            avg_diagonal_quality = diagonal_angle_quality / diagonal_count if diagonal_count > 0 else 0.0

            # More permissive thresholds for zigzag detection
            if diagonal_count > 8 and diagonal_count / total_lines > 0.15:  # Lower thresholds
                line_count_score = min(1.0, diagonal_count / 15.0)  # Lower threshold for line count
                angle_quality_score = avg_diagonal_quality
                confidence = 0.6 * line_count_score + 0.4 * angle_quality_score  # + 0.1 * length_similarity

                return True, confidence

        return False, 0.0


class PositionClassifier(ConceptClassifier):
    """
    Classifier for detecting the position of shapes in generated images.
    """

    def __init__(self):
        self.positions = tuple(_constants.CONCEPT_TO_SYNTHETIC_MAP["position"].keys())
        self.hsv_threshold = 100
        self.min_area = 500
        self.max_area = 50000
        self.tolerance = 0.05

    def classify(self, image: np.ndarray) -> str | None:
        """
        Main function to classify the position of the shape in an image

        Args:
            image: numpy array (RGB format)
        """
        # Convert RGB to BGR (OpenCV uses BGR)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        _, gray = cv2.threshold(v_channel, self.hsv_threshold, 255, cv2.THRESH_BINARY_INV)

        # Clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        binary_clean = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel)
        # Add dilation to fill small gaps and make shape more solid
        binary_clean = cv2.dilate(binary_clean, kernel, iterations=1)

        # Split image into 4 quadrants
        image_h, image_w = gray.shape
        mid_h, mid_w = image_h // 2, image_w // 2

        # Define the 4 quadrants
        quadrants = [
            gray[0:mid_h, 0:mid_w],  # Top-left
            gray[0:mid_h, mid_w:image_w],  # Top-right
            gray[mid_h:image_h, 0:mid_w],  # Bottom-left
            gray[mid_h:image_h, mid_w:image_w],  # Bottom-right
        ]

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_clean, connectivity=8)

        if num_labels <= 1:
            return None

        # Filter components by size
        valid_components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_area <= area <= self.max_area:
                centroid = centroids[i]
                valid_components.append((i, area, centroid))

        if len(valid_components) == 0 or len(valid_components) > 1:
            return None

        component_label, component_area, centroid = valid_components[0]
        center_x, center_y = centroid

        # Get bounding box of the component
        x = stats[component_label, cv2.CC_STAT_LEFT]
        y = stats[component_label, cv2.CC_STAT_TOP]
        w = stats[component_label, cv2.CC_STAT_WIDTH]
        h = stats[component_label, cv2.CC_STAT_HEIGHT]

        # Calculate bounding box corners
        left = x
        right = x + w
        top = y
        bottom = y + h

        # Allow 5% tolerance for shapes to extend into adjacent quadrants
        tolerance_w = image_w * self.tolerance
        tolerance_h = image_h * self.tolerance

        # Check which quadrant the shape primarily fits in (with 5% tolerance)
        fits_top_left = (right <= mid_w + tolerance_w) and (bottom <= mid_h + tolerance_h)
        fits_top_right = (left >= mid_w - tolerance_w) and (bottom <= mid_h + tolerance_h)
        fits_bottom_left = (right <= mid_w + tolerance_w) and (top >= mid_h - tolerance_h)
        fits_bottom_right = (left >= mid_w - tolerance_w) and (top >= mid_h - tolerance_h)

        if fits_top_left:
            detected_position = "top left"
        elif fits_top_right:
            detected_position = "top right"
        elif fits_bottom_left:
            detected_position = "bottom left"
        elif fits_bottom_right:
            detected_position = "bottom right"
        else:
            return None

        assert detected_position in self.positions, "Detected position is not in the list of positions"
        return detected_position


class ColorClassifier(ConceptClassifier):
    """
    Classifier for detecting the most prominent color in generated images.
    """

    def __init__(self):
        self.colors = tuple(_constants.CONCEPT_TO_SYNTHETIC_MAP["color"].keys())
        # Define HSV hue ranges for each color
        self.color_ranges = {
            'red': [(0, 15), (166, 180)],
            'green': [(46, 75)],
            'blue': [(106, 135)],
            'yellow': [(16, 45)],
            'purple': [(136, 165)],
            'turquoise': [(76, 105)]
        }

        # Thresholds for filtering
        self.min_saturation = 50  # S threshold
        self.min_value = 50       # V threshold
        self.min_pixel_count = 30  # Minimum pixels to consider a color

    def classify(self, image: np.ndarray) -> str | None:
        """
        Main function to classify the most prominent color in an image

        Args:
            image: numpy array (RGB format)
        """

        assert image.ndim == 3 and image.shape[2] == 3, "Image must be in RGB format"

        # Convert RGB to BGR (OpenCV uses BGR)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Convert BGR to HSV for better color detection
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Extract H, S, V channels
        h_channel = hsv[:, :, 0]  # Hue (0-179)
        s_channel = hsv[:, :, 1]  # Saturation (0-255)
        v_channel = hsv[:, :, 2]  # Value (0-255)

        # Count pixels for each color
        color_counts = {}

        for color_name, ranges in self.color_ranges.items():
            count = 0

            for h_min, h_max in ranges:
                # Create mask for hue range
                if h_min <= h_max:
                    # Normal range (e.g., 46-75)
                    hue_mask = (h_channel >= h_min) & (h_channel <= h_max)
                else:
                    # Wrapped range (e.g., red: 0-15 or 166-180)
                    hue_mask = (h_channel >= h_min) | (h_channel <= h_max)

                # Apply saturation and value filters
                saturation_mask = s_channel >= self.min_saturation
                value_mask = v_channel >= self.min_value

                # Combine all masks
                combined_mask = hue_mask & saturation_mask & value_mask

                # Count pixels
                count += np.sum(combined_mask)

            color_counts[color_name] = count

        # Find the color with the most pixels
        if len(color_counts) == 0:
            return None  # No colors detected

        if not color_counts:
            return None

        max_color = max(color_counts, key=color_counts.get)
        max_count = color_counts[max_color]

        if max_count < self.min_pixel_count:
            return None

        assert max_color in self.colors, "Max color is not in the list of colors"
        return max_color

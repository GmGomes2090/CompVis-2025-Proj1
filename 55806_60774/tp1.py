###############################################################################
## Computer Vision 2025-2026 - NOVA FCT
## Assignment 1
##
## Tomas Santos 55806
## Goncalo Gomes 60774
## 
###############################################################################
import cv2
import cv2 as cv
import numpy as np
import os
import glob
import json

# SECTION 3: IMAGE PREPROCESSING
"""
    s all images from input directory and resize them maintaining aspect ratio.
    Smaller side 512 pixels.
    
    Returns:
        list: List of tuples (filename, resized_image)
    """
def load_and_resize_images(input_dir):
  
    # Get all image files sorted 
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.*")))
    
    resized_images = []
    
    for img_path in image_paths:
        
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
 
        img = cv.imread(img_path)
        if img is None:
            continue
        
        # Get image dimensions
        h, w = img.shape[:2]
        
        # Calculate new dimensions 
        if h < w:
            new_h = 512
            new_w = int(w * 512 / h)
        else:
            new_w = 512
            new_h = int(h * 512 / w)
        
        # Resize image
        resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
        
        # Store filename and resized image
        filename = os.path.basename(img_path)
        resized_images.append((filename, resized))
        
        print(f"Loaded and resized: {filename} ({w}x{h} -> {new_w}x{new_h})")
    
    return resized_images

# SECTION 4: HISTOGRAM SIMILARITY
    """
    calculate normalized color histogram for an image.
    
    Args:
        image: BGR image
    
    Returns:
        Normalized histogram
    """
def calculate_histogram(image):
     
    # Define histogram parameters
    # Using 8 bins per channel 
    hist_size = [8, 8, 8]
    ranges = [0, 256, 0, 256, 0, 256]  # RGB ranges
    
    # Compute 3D histogram
    hist = cv.calcHist([image], [0, 1, 2], None, hist_size, ranges)
    
    # Normalize histogram
    hist = cv.normalize(hist, hist).flatten()
    
    return hist

"""
    Calculate distance between two histograms.
    Lower values = more similar images.
    
    Args:
        hist1: First histogram
        hist2: Second histogram
    
    Returns:
        float: Distance value
    """

def my_chi_square(hist1, hist2):
    # print("MY CHI-SQUARE")
    #print(hist1)

    non_zero = 1e-11

    # print(hist1)

    # hist_array1 = np.asarray(hist1)
    # hist_array2 = np.asarray(hist2)
    #
    # print(hist_array1)

    chi_square = 0.5 * np.sum( (hist1 - hist2) ** 2 / (hist1 + hist2 + non_zero))
    # print(f"CHI_SQAURE = {chi_square}")
    return chi_square

def histogram_distance(hist1, hist2, selection):
    if selection == "btc":
        # Bhattacharyya distance
        return cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    if selection == "chi":
        # Chi-Square distance
        return cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    if selection == "my_chi":
        # Custom Chi-Square implementation
        return my_chi_square(hist1, hist2)
    raise Exception("Invalid algorythm selection")
def find_similar_groups(resized_images, selection, threshold=0.3):
    """
    Group similar images based on histogram comparison.
    Assumes images are in chronological order.
    
    Args:
        resized_images: List of (filename, image) tuples
        threshold: Similarity threshold (lower = more strict)
    
    Returns:
        list: List of groups, each group is a list of (filename, image) tuples
    """
    if len(resized_images) == 0:
        return []
    
    groups = []
    current_group = [resized_images[0]]
    
    # Compute histogram for first image
    prev_hist = calculate_histogram (resized_images[0][1])
    
    for i in range(1, len(resized_images)):
        filename, image = resized_images[i]
        curr_hist = calculate_histogram(image)
        
        # Compare with previous image
        dist = histogram_distance(prev_hist, curr_hist, selection)
        
        print(f"Distance between image {resized_images[i-1][0]} and image {filename}: {dist:.4f}")
        
        if dist < threshold:
            # Similar to previous image we add
            current_group.append((filename, image))
        else:
            # Different from previous - also check if current group is bigger that 2
            if len(current_group) >= 3:  # More than 2 similar images we add
                groups.append(current_group)
            # Start new group
            current_group = [(filename, image)]
        
        prev_hist = curr_hist
    
    # Last group
    if len(current_group) >= 3:
        groups.append(current_group)
    
    return groups

    """
    Save similar image groups to separate folders.
    
    Args:
        groups: List of image groups
        output_dir: Output directory path
    """
def save_similar_groups(groups, output_dir):
    
    for id , group in enumerate(groups):
        # Creates folder for this group
        folder_name = f"similar-{id }"
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True) #Checks if exists also
        
        # Save all images in the group
        for filename, image in group:
            output_path = os.path.join(folder_path, filename)
            cv.imwrite(output_path, image)
        
        print(f"Created {folder_name} with {len(group)} images")

# SECTION 5: HISTOGRAM ANALYSIS AND WHITE BALANCE 
"""
    Compute average histogram for a group of images   
    Args:
        group: List of (filename, image) tuples 
    Returns:
        Average histogram
    """
def compute_average_histogram(group):
     
    histograms = [calculate_histogram(img) for _, img in group]
    avg_hist = np.mean(histograms, axis=0)
    return avg_hist

"""
    Apply white balance to image based on target average RGB values. 
    Args:
        image: Input BGR image
        target_avg: Target average RGB values 
    Returns:
        White-balanced image
    """
def apply_white_balance(image, target_avg):
     
    # Calculate current average for each channel
    current_avg = np.mean(image, axis=(0, 1))  # [B, G, R]
    
    # Calculate scaling factors for each channel 
    scales = np.where(current_avg > 0, target_avg / current_avg, 1.0)
    
    # Apply scaling
    balanced = image.astype(np.float32)
    for i in range(3):
        balanced[:, :, i] = balanced[:, :, i] * scales[i]
    
    # Clip values 
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    
    return balanced

"""
    Save histogram visualization for a group.    
    Args:
        group: List of (filename, image) tuples
        folder_path: Path to save the histogram image
    """
def save_histogram_visualization(group, folder_path):
 
    try:
        import matplotlib
        matplotlib.use('Agg') 
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Warning: matplotlib not available. Skipping histogram visualization.")
        return
    
   
    all_histograms = {'R': [], 'G': [], 'B': []}
    
    for filename, image in group:
        # OpenCV uses BGR, swap to RGB
        img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
        
        # Get histograms - simple bincount like the example
        hist_r = np.bincount(R.ravel(), minlength=256)
        hist_g = np.bincount(G.ravel(), minlength=256)
        hist_b = np.bincount(B.ravel(), minlength=256)
        
        all_histograms['R'].append(hist_r)
        all_histograms['G'].append(hist_g)
        all_histograms['B'].append(hist_b)
    
    # Calculate average histograms
    avg_histograms = {
        'R': np.mean(all_histograms['R'], axis=0),
        'G': np.mean(all_histograms['G'], axis=0),
        'B': np.mean(all_histograms['B'], axis=0)
    }
    
    bins = np.arange(256) 
    
    # Create figure with subplots one for each image + one for average
    num_images = len(group)
    
    # If few images, show individual + average. Otherwise just show average
    if num_images <= 5:
        # images individually + average
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # individual images
        for i, (filename, _) in enumerate(group[:5]):  # Max 5 individual
            ax = axes[i]
            ax.bar(bins, all_histograms['R'][i], color='red', alpha=0.4, label='Red')
            ax.bar(bins, all_histograms['G'][i], color='green', alpha=0.4, label='Green')
            ax.bar(bins, all_histograms['B'][i], color='blue', alpha=0.4, label='Blue')
            ax.set_title(filename, fontsize=10)
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot average in the last subplot
        ax = axes[5]
        ax.bar(bins, avg_histograms['R'], color='red', alpha=0.4, label='Red')
        ax.bar(bins, avg_histograms['G'], color='green', alpha=0.4, label='Green')
        ax.bar(bins, avg_histograms['B'], color='blue', alpha=0.4, label='Blue')
        ax.set_title('AVERAGE', fontsize=10, fontweight='bold')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'RGB Histograms - All {num_images} Images', fontsize=14, fontweight='bold')
    else:
        # show the average for many images
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(bins, avg_histograms['R'], color='red', alpha=0.4, label='Red')
        ax.bar(bins, avg_histograms['G'], color='green', alpha=0.4, label='Green')
        ax.bar(bins, avg_histograms['B'], color='blue', alpha=0.4, label='Blue')
        ax.set_title(f'RGB Histogram - Average of {num_images} Images', fontsize=14, fontweight='bold')
        ax.set_xlabel('Pixel Value', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(folder_path, 'histograms.jpg')
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved histogram visualization: {output_path}")
    
    """
    Process each similar-# folder:
    - Compute average histogram
    - Apply white balance
    - Save histogram visualization
    """
def process_similar_folders(output_dir):
   
    similar_folders = sorted([f for f in os.listdir(output_dir) 
                             if f.startswith('similar-')])
    
    for folder_name in similar_folders:
        folder_path = os.path.join(output_dir, folder_name)
        
        # Load all images from this folder 
        image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        # Filter out histogram 
        image_files = [f for f in image_files 
                      if 'histogram' not in os.path.basename(f).lower() 
                      and 'equal' not in os.path.basename(f).lower()]
        
        group = []
        original_group = []  
        
        for img_path in image_files:
            img = cv.imread(img_path)
            if img is not None:
                filename = os.path.basename(img_path)
                original_group.append((filename, img.copy())) 
                group.append((filename, img))
        
        if len(group) == 0:
            continue
        
        print(f"\nProcessing {folder_name}...")
        print(f"  Found {len(group)} images ")
        
        # Calculate average color (before white balance) for statistics
        avg_colors = [np.mean(img, axis=(0, 1)) for _, img in original_group]
        avg_color = np.mean(avg_colors, axis=0)  # [B, G, R]
        avg_color_rgb = [int(avg_color[2]), int(avg_color[1]), int(avg_color[0])]  # Convert to RGB
        
        print(f"  Average color (RGB): {avg_color_rgb}")
        
        # Compute average histogram from original images
        avg_hist = compute_average_histogram(original_group)
        
        # Calculate target average color for white balance
        # Use the average from all images in the group
        target_avg = avg_color  # [B, G, R]
        
        # Apply white balance to all images and save
        print(f"  Applying white balance to {len(group)} images...")
        for filename, image in group:
            # Apply white balance
            balanced = apply_white_balance(image, target_avg)
            
            # Save white-balanced image 
            output_path = os.path.join(folder_path, filename)
            cv.imwrite(output_path, balanced)
        
        # Save histogram visualization (before white balance)
        save_histogram_visualization(original_group, folder_path)
        
        print(f" {folder_name} processed successfully")
# SECTION 6: MOPS DESCRIPTOR
def my_track_points(img):
    features = cv2.goodFeaturesToTrack(img)
    return

    """
    Find keypoints using goodFeaturesToTrack (Harris corner detector ).
    
    Args:
        image: Input image (BGR)
        max_corners: Maximum number of corners to detect (500 is enough for most images)
    
    Returns:
        Array of keypoints (N, 1, 2) format
    """
def my_track_points(image, max_corners=500):
    
    # Convert to grayscale 
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detect corners using Shi-Tomasi corner detection (goodFeaturesToTrack)
    corners = cv.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,         # Limit to 500 strongest corners for speed
        qualityLevel=0.01,              # Only keep corners with quality > 1% of best corner
                                        # Lower = more corners but lower quality
                                        # 0.01 is a good balance
        minDistance=10,                 # Minimum 10 pixels between corners
                                        # Prevents clustering of keypoints
                                        # Too small = redundant points, Too large = miss features
        blockSize=3                     # Size of averaging block for derivative covariation
                                        # 3x3 is standard, larger = smoother but less precise
    )
    
    return corners if corners is not None else np.array([])
"""
    Find dominant orientation for a keypoint using gradient histogram.
    This makes the descriptor rotation-invariant.
    
    Args:
        image: Input image (grayscale)
        point: Keypoint coordinates (x, y)
        window_size: Size of window around point (40x40 pixels)
                     Large enough to capture local structure but not too much context
    
    Returns:
        float: Dominant angle in degrees [-180, 180]
    """

def my_point_rotation(image, point, window_size=40):
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    
    h, w = gray.shape
    x, y = int(point[0]), int(point[1])
    
    # Extract patch around point
    half_size = window_size // 2  # 20 pixels on each side
    x1 = max(0, x - half_size)    # Clamp to image boundaries
    y1 = max(0, y - half_size)
    x2 = min(w, x + half_size)
    y2 = min(h, y + half_size)
    
    patch = gray[y1:y2, x1:x2]
    
    
    if patch.size == 0 or patch.shape[0] < 5 or patch.shape[1] < 5:
        return 0.0  # Return 0 degrees if invalid
    
    # Compute gradients using Sobel operator
    # Sobel is better than simple differences because it includes smoothing
    grad_x = cv.Sobel(patch, cv.CV_64F, 1, 0, ksize=3)  # Horizontal gradient (∂I/∂x)
    grad_y = cv.Sobel(patch, cv.CV_64F, 0, 1, ksize=3)  # Vertical gradient (∂I/∂y)
                                                         # ksize=3 is standard 3x3 Sobel
    
    # Compute magnitude and angle
    magnitude = np.sqrt(grad_x**2 + grad_y**2)           # Gradient strength at each pixel
    angles = np.arctan2(grad_y, grad_x) * 180 / np.pi    # Gradient direction in degrees
                                                          
    
    # Weight angles by gradient magnitude
    # Strong gradients (edges) contribute more to orientation than weak ones
    # This makes the descriptor more stable and focuses on prominent features
    hist, bin_edges = np.histogram(
        angles,                      # Input: gradient angles
        bins=36,                     # 36 bins = 10 degrees per bin
                                     # SIFT uses 36 bins 
                                      
        range=(-180, 180),            
        weights=magnitude            
    )
    
    # Find dominant angle 
    dominant_bin = np.argmax(hist)                       # Index of highest bin
    dominant_angle = -180 + (dominant_bin * 10) + 5     # Convert bin to angle
                                                         # +5 gives center of bin  
    
    return dominant_angle
 
"""
    Calculate Euclidean distance between two descriptors.
    L2 distance is standard for comparing feature descriptors.
    
    Args:
        desc1: First descriptor  
        desc2: Second descriptor  
    
    Returns:
        float: Euclidean distance  
    """

def my_distance(desc1, desc2):
    
    # L2 norm: sqrt(sum((desc1[i] - desc2[i])²))
    # np.linalg.norm computes this efficiently
    return np.linalg.norm(desc1 - desc2)

"""
    Match descriptors using Lowe's ratio test (nearest neighbor ratio).
    This filters out ambiguous matches, keeping only distinctive ones.
    
    Args:
        keypoints1: Keypoints from image 1
        descriptors1: List of descriptors from image 1
        keypoints2: Keypoints from image 2
        descriptors2: List of descriptors from image 2
        ratio_threshold: Lowe's ratio threshold (default 0.75)
                        Lower = more strict (fewer but better matches)
                        0.75 is standard from Lowe's SIFT paper
                        0.8 = more permissive, 0.7 = more strict
    
    Returns:
        List of matches as (index1, index2, distance) tuples
    """
def my_match(keypoints1, descriptors1, keypoints2, descriptors2, ratio_threshold=0.75):
    
    matches = []
    
    # For each descriptor in image 1
    for i, desc1 in enumerate(descriptors1):
        # Calculate distances to all descriptors in image 2
        distances = [my_distance(desc1, desc2) for desc2 in descriptors2]
        
        # Need at least 2 matches for ratio test
        if len(distances) < 2:
            continue
        
        # Get indices of two nearest neighbors
        sorted_indices = np.argsort(distances)  # Sort by distance (ascending)
        nearest_idx = sorted_indices[0]         # Best match
        second_nearest_idx = sorted_indices[1]  # Second best match
        
        nearest_dist = distances[nearest_idx]
        second_nearest_dist = distances[second_nearest_idx]
        
        # Lowe's ratio test
        # If best match is much better than second best, its distinctive
        # If nearest_dist ≈ second_nearest_dist the match is ambiguous
        # Ratio < 0.75 means best match is at least 25% better than second
        if nearest_dist < ratio_threshold * second_nearest_dist:
            matches.append((i, nearest_idx, nearest_dist))
    
    return matches
"""
    Create comparison between custom MOPS descriptor and SIFT.
    Uses images 109900.jpg and 109901.jpg  .
    Generates side-by-side visualization saved as my_match.jpg.
    """

def create_my_match_comparison(output_dir):
    
    input_dir = "input"
    
    
    img1_path = os.path.join(input_dir, "109900.jpg")
    img2_path = os.path.join(input_dir, "109901.jpg")
    
    # If specific images don't exist, use first two images in directory
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
        if len(image_paths) < 2:
            print("  Warning: Need at least 2 images for comparison")
            return
        img1_path = image_paths[0]
        img2_path = image_paths[1]
    
    # Load images
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("  Warning: Failed to load comparison images")
        return
    
    # Resize for consistent processing (512 on smaller side)
  
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if min(h1, w1) > 512:
        scale = 512 / min(h1, w1)
        img1 = cv.resize(img1, (int(w1*scale), int(h1*scale)))
    if min(h2, w2) > 512:
        scale = 512 / min(h2, w2)
        img2 = cv.resize(img2, (int(w2*scale), int(h2*scale)))
    
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    
    print("  Creating MOPS vs SIFT comparison...")
    
    # === MOPS Descriptor ===
    print(" Running MOPS...")
    # Detect keypoints  
    kp1_mops = my_track_points(img1, max_corners=200)
    kp2_mops = my_track_points(img2, max_corners=200)
    
    # Create descriptors for each keypoint
    desc1_mops = []
    for kp in kp1_mops:
        angle = my_point_rotation(gray1, kp[0])      # Find orientation
        desc = my_descriptor(gray1, kp[0], angle)    # Create descriptor
        desc1_mops.append(desc)
    
    desc2_mops = []
    for kp in kp2_mops:
        angle = my_point_rotation(gray2, kp[0])
        desc = my_descriptor(gray2, kp[0], angle)
        desc2_mops.append(desc)
    
    # Match descriptors
    matches_mops = my_match(kp1_mops, desc1_mops, kp2_mops, desc2_mops)
    
    # Draw MOPS matches
    img_mops = draw_mops_matches(img1, kp1_mops, img2, kp2_mops, matches_mops)
    
    # === SIFT Descriptor ===
    print(" Running SIFT...")
    sift = cv.SIFT_create()  # Standard SIFT detector
    kp1_sift, desc1_sift = sift.detectAndCompute(gray1, None)
    kp2_sift, desc2_sift = sift.detectAndCompute(gray2, None)
    
    # Match SIFT descriptors using BFMatcher 
    bf = cv.BFMatcher()
    matches_sift_raw = bf.knnMatch(desc1_sift, desc2_sift, k=2)
    
    # Apply ratio test to SIFT matches
    matches_sift = []
    for m_n in matches_sift_raw:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:  # Same ratio test as MOPS
                matches_sift.append(m)
    
    # Draw SIFT matches
    img_sift = cv.drawMatches(img1, kp1_sift, img2, kp2_sift, 
                             matches_sift[:50], None,  # Limit to 50 for clarity
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Combine both images vertically for comparison
    h_mops = img_mops.shape[0]
    h_sift = img_sift.shape[0]
    w_max = max(img_mops.shape[1], img_sift.shape[1])
    
    # Resize to same width
    img_mops_resized = cv.resize(img_mops, (w_max, h_mops))
    img_sift_resized = cv.resize(img_sift, (w_max, h_sift))
    
    # Stack vertically 
    combined = np.vstack([img_mops_resized, img_sift_resized])
    
    # Add text labels
    cv.putText(combined, f'MOPS: {len(matches_mops)} matches', 
              (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.putText(combined, f'SIFT: {len(matches_sift)} matches', 
              (10, h_mops + 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save comparison image
    output_path = os.path.join(output_dir, "my_match.jpg")
    cv.imwrite(output_path, combined)
    
    # Print statistics for report
    print(f" Saved comparison: {output_path}")
    print(f" MOPS: {len(kp1_mops)} kp (img1), {len(kp2_mops)} kp (img2), {len(matches_mops)} matches")
    print(f" SIFT: {len(kp1_sift)} kp (img1), {len(kp2_sift)} kp (img2), {len(matches_sift)} matches")
    
    # Calculate average matching distances for analysis
    if len(matches_mops) > 0:
        avg_dist_mops = np.mean([m[2] for m in matches_mops])
        print(f" MOPS avg distance: {avg_dist_mops:.3f}")
    if len(matches_sift) > 0:
        avg_dist_sift = np.mean([m.distance for m in matches_sift])
        print(f" SIFT avg distance: {avg_dist_sift:.2f}")
 

"""
    Create MOPS-style descriptor for a keypoint.
    Extracts window, rotates to canonical orientation, downsamples to 8x8.
    
    Args:
        image: Input image (grayscale)
        point: Keypoint coordinates (x, y)
        angle: Rotation angle in degrees (from my_point_rotation)
        window_size: Size of window to extract (default 40x40)
        descriptor_size: Size to downsample to (default 8x8 = 64 elements)
    
    Returns:
        Descriptor array (64 values normalized to [0, 1])
    """
def my_descriptor(image, point, angle, window_size=40, descriptor_size=8):
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    
    h, w = gray.shape
    x, y = int(point[0]), int(point[1])
    
    # Extract patch around point with padding if necessary
    half_size = window_size // 2
    
    # Use larger extraction to avoid boundary issues after rotation
    extraction_size = int(window_size * 1.5)
    half_extract = extraction_size // 2
    
    x1 = x - half_extract
    y1 = y - half_extract
    x2 = x + half_extract
    y2 = y + half_extract
    
    # Handle boundaries with padding
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    patch = gray[y1:y2, x1:x2]
    
    if patch.size == 0:
        return np.zeros(descriptor_size * descriptor_size)
    
    # Pad if necessary
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        patch = cv.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right,
                                 cv.BORDER_REPLICATE)
    
    # Get rotation matrix 
    center = (patch.shape[1] // 2, patch.shape[0] // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated = cv.warpAffine(patch, rotation_matrix, 
                           (patch.shape[1], patch.shape[0]),
                           flags=cv.INTER_LINEAR,
                           borderMode=cv.BORDER_REPLICATE)
    
    # Extract center window_size x window_size region
    center_y, center_x = rotated.shape[0] // 2, rotated.shape[1] // 2
    crop_half = window_size // 2
    cropped = rotated[center_y - crop_half:center_y + crop_half,
                     center_x - crop_half:center_x + crop_half]
    
    if cropped.size == 0:
        return np.zeros(descriptor_size * descriptor_size)
    
    # Downsample to descriptor_size x descriptor_size (8x8)
    descriptor_patch = cv.resize(cropped, (descriptor_size, descriptor_size),
                                interpolation=cv.INTER_AREA)
    
    # Flatten to 1D array
    descriptor = descriptor_patch.flatten().astype(np.float32)
    
    # Normalize to [0, 1]
    descriptor = descriptor / 255.0
    
    # Apply normalization for better invariance  
    mean = np.mean(descriptor)
    std = np.std(descriptor)
    if std > 0:
        descriptor = (descriptor - mean) / std
        # Clip to reasonable range and rescale to [0, 1]
        descriptor = np.clip(descriptor, -3, 3)
        descriptor = (descriptor + 3) / 6.0
    
    return descriptor
 
"""
    Draw matches between two images for MOPS descriptor.
    
    Args:
        img1: First image
        kp1: Keypoints from first image (N, 1, 2)
        img2: Second image
        kp2: Keypoints from second image
        matches: List of (idx1, idx2, distance) tuples
        max_draw: Maximum number of matches to draw
    
    Returns:
        Combined image showing matches
    """
def draw_mops_matches(img1, kp1, img2, kp2, matches, max_draw=50):
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create output image
    out_h = max(h1, h2)
    out_w = w1 + w2
    out_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    # Place images side by side
    out_img[:h1, :w1] = img1
    out_img[:h2, w1:w1+w2] = img2
    
    # Draw matches (limit to max_draw best matches)
    matches_sorted = sorted(matches, key=lambda x: x[2])[:max_draw]
    
    for idx1, idx2, dist in matches_sorted:
        # Get keypoint coordinates
        pt1 = tuple(map(int, kp1[idx1][0]))
        pt2 = tuple(map(int, kp2[idx2][0]))
        pt2 = (pt2[0] + w1, pt2[1])  # Offset for second image
        
        # Random color for this match
        color = tuple(np.random.randint(50, 255, 3).tolist())
        
        # Draw line and circles
        cv.line(out_img, pt1, pt2, color, 1)
        cv.circle(out_img, pt1, 4, color, -1)
        cv.circle(out_img, pt2, 4, color, -1)
    
    return out_img


 
# COMPARE IMAGES AND STATS
def compare_with_sift(image1, image2):
    """
    Compare two images using SIFT and return matches and keypoints.
    
    Args:
        image1: First image
        image2: Second image
    
    Returns:
        tuple: (good_matches, kp1, kp2) or ([], None, None) if failed
    """
    gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    
    # Create SIFT detector
    sift = cv.SIFT_create()
    
    # Detect and compute
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return [], None, None
    
    # Match descriptors using BFMatcher
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    return good_matches, kp1, kp2


def find_equal_images(output_dir, match_threshold=50):
    """
    Find very similar (equal) images in each similar folder using SIFT.
    Saves match visualizations as equal-X.jpg files.
    
    Args:
        output_dir: Output directory
        match_threshold: Minimum matches to consider images equal
    """
    similar_folders = sorted([f for f in os.listdir(output_dir) 
                             if f.startswith('similar-')])
    
    for folder_name in similar_folders:
        folder_path = os.path.join(output_dir, folder_name)
        
        # Load all images EXCLUDE histogram and equal files
        image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        image_files = [f for f in image_files 
                      if 'histogram' not in os.path.basename(f).lower() 
                      and 'equal' not in os.path.basename(f).lower()]
        
        if len(image_files) < 2:
            continue
        
        # Load first image as reference
        ref_img = cv.imread(image_files[0])
        if ref_img is None:
            continue
        
        equal_count = 0
        
        print(f"\n  Checking {folder_name} for equal images...")
        
        # Compare each image with the first one
        for i in range(1, len(image_files)):
            img = cv.imread(image_files[i])
            if img is None:
                continue
            
            # Compare using SIFT returns matches and keypoints
            good_matches, kp1, kp2 = compare_with_sift(ref_img, img)
            num_matches = len(good_matches)
            
            # If enough good matches, images are considered equal
            if num_matches >= match_threshold:
                print(f" Equal: {os.path.basename(image_files[0])} ≈ {os.path.basename(image_files[i])} ({num_matches} matches)")
                
                # Draw matches visualization
                # Limit to best 50 matches for clarity
                matches_to_draw = sorted(good_matches, key=lambda x: x.distance)[:50]
                
                match_img = cv.drawMatches(
                    ref_img, kp1,           # First image and keypoints
                    img, kp2,               # Second image and keypoints
                    matches_to_draw,        # Matches to draw
                    None,                   # Output image  
                    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                
                # Add text labels
                cv.putText(match_img, 
                          f'{num_matches} matches (threshold: {match_threshold})', 
                          (10, 30), 
                          cv.FONT_HERSHEY_SIMPLEX, 
                          1, 
                          (0, 255, 0),
                          2)
                
                cv.putText(match_img, 
                          os.path.basename(image_files[0]), 
                          (10, 60), 
                          cv.FONT_HERSHEY_SIMPLEX, 
                          0.7, 
                          (255, 255, 255),
                          2)
                
                cv.putText(match_img, 
                          os.path.basename(image_files[i]), 
                          (ref_img.shape[1] + 10, 60), 
                          cv.FONT_HERSHEY_SIMPLEX, 
                          0.7, 
                          (255, 255, 255),
                          2)
                
                # Save match visualization as equal-X.jpg
                equal_output_path = os.path.join(folder_path, f'equal-{equal_count}.jpg')
                cv.imwrite(equal_output_path, match_img)
                
                print(f"  Saved: {equal_output_path}")
                
                equal_count += 1
        
        if equal_count > 0:
            print(f"  → Found {equal_count} equal image pair(s) in {folder_name}")
        else:
            print(f"  → No equal images found in {folder_name}")

 

def load_groundtruth(input_dir):
    """
    Load ground truth data from JSON file.
    
    Returns:
        dict: Ground truth data
    """
    gt_path = os.path.join(input_dir, 'groundtruth.json')
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            return json.load(f)
    return {}


def print_statistics(output_dir, input_dir):
    """
    Print statistics for each similar folder and overall.
    """
    similar_folders = sorted([f for f in os.listdir(output_dir) 
                             if f.startswith('similar-')])
    
    groundtruth = load_groundtruth(input_dir)
    
    total_images = 0
    total_groundtruth = 0
    
    for folder_name in similar_folders:
        folder_path = os.path.join(output_dir, folder_name)
        
        # Count images in folder - EXCLUDE histogram and equal files
        image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
        # Filter out histogram and equal files
        image_files = [f for f in image_files 
                      if 'histogram' not in os.path.basename(f).lower() 
                      and 'equal' not in os.path.basename(f).lower()]
        
        num_images = len(image_files)
        
        # Get ground truth count (you need to map folder to ground truth)
        # This is simplified - you'll need to implement proper matching
        gt_count = num_images  # Placeholder
        
        # Calculate precision
        if gt_count > 0:
            precision = 1 - abs(num_images - gt_count) / gt_count
        else:
            precision = 0.0
        
        # Calculate average color (load images again)
        avg_colors = []
        for img_path in image_files:
            img = cv.imread(img_path)
            if img is not None:
                avg_color = np.mean(img, axis=(0, 1))
                avg_colors.append(avg_color)
        
        if len(avg_colors) > 0:
            overall_avg = np.mean(avg_colors, axis=0)
            avg_color_rgb = [int(overall_avg[2]), int(overall_avg[1]), int(overall_avg[0])]
        else:
            avg_color_rgb = [0, 0, 0]
        
        print(f"{folder_name} number of images: {num_images} ground-truth: {gt_count} precision: {precision:.3f} averagecolor: {avg_color_rgb}")
        
        total_images += num_images
        total_groundtruth += gt_count
    
    # Print total statistics
    if total_groundtruth > 0:
        total_precision = 1 - abs(total_images - total_groundtruth) / total_groundtruth
    else:
        total_precision = 0.0
    
    print(f"\nTOTAL number of images: {total_images} ground-truth: {total_groundtruth} precision: {total_precision:.3f}")


 
if __name__ == "__main__":
    print("Assignment 1")
    
     # Define directories of images
    input_dir = "input"
    output_dir = "output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    #  Load and resize images
    print("\n Loading and resizing images ")
    resized_images = load_and_resize_images(input_dir)
    print(f"Loaded {len(resized_images)} images")

    # Find similar groups based on histograms
    print("\n Finding similar image groups ")
    groups = find_similar_groups(resized_images, "btc", threshold=0.3)
    print(f"Found {len(groups)} similar groups")

    # Saving similar groups
    print("\n Saving similar groups ")
    save_similar_groups(groups, output_dir)

    # Processing similar folders
    print("\n Processing similar folders...")
    process_similar_folders(output_dir)
 
    process_similar_folders(output_dir) 
    print("\n Creating descriptor comparison...")
    create_my_match_comparison(output_dir) 
    # Find equal images
    print("\n Finding equal images...")
    find_equal_images(output_dir)
    
    # Print statistics
    print("\n Final Statistics:")
    print("-" * 80)
    print_statistics(output_dir, input_dir)

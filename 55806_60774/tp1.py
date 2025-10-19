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
    print("MY CHI-SQUARE")
    #print(hist1)

    non_zero = 1e-9

    hist_array1 = np.asarray(hist1)
    hist_array2 = np.asarray(hist2)

    chi_square = 0.5 * np.sum( (hist_array1 - hist_array2) ** 2 / (hist_array1 + hist_array2 + non_zero))
    print(f"CHI_SQAURE = {chi_square}")
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


 
if __name__ == "__main__":
    print("Assignment 1")
    
     # Define directories of images
    input_dir = "../input"
    output_dir = "../output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    #  Load and resize images
    print("\n Loading and resizing images ")
    resized_images = load_and_resize_images(input_dir)
    print(f"Loaded {len(resized_images)} images")

    # Find similar groups based on histograms
    print("\n Finding similar image groups ")
    groups = find_similar_groups(resized_images, "chi", threshold=0.3)
    print(f"Found {len(groups)} similar groups")

    # Saving similar groups
    print("\n Saving similar groups ")
    save_similar_groups(groups, output_dir)

    # Processing similar folders
    print("\n Processing similar folders...")
    process_similar_folders(output_dir)
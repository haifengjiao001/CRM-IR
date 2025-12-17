import torch
import numpy as np
import cv2 # Make sure OpenCV is installed (pip install opencv-python)
def normalize(value, min_value, max_value):
    """
    将一个值归一化到0到1的范围

    :param value: 原始值
    :param min_value: 最小值
    :param max_value: 最大值
    :return: 归一化后的值
    """
    return (value - min_value) / (max_value - min_value)
def save_rgb_tensor_as_16bit_png_cv2(tensor: torch.Tensor, output_path: str):
    """
    Saves a (1, 3, H, W) float32 PyTorch tensor (RGB order) as a 16-bit PNG image.
    Assumes tensor values are in the range [0.0, 1.0].

    Args:
        tensor (torch.Tensor): Input tensor with shape (1, 3, H, W), dtype float32, RGB order.
        output_path (str): Path to save the PNG image (e.g., "output_image.png").
    """
    if not output_path.lower().endswith(".png"):
        # Append .png if not already present or a different extension is used
        base, ext = os.path.splitext(output_path)
        output_path = base + ".png"
        print(f"Output path set to: {output_path}")


    # 1. Validate tensor shape and dtype
    if not (tensor.ndim == 4 and tensor.shape[0] == 1 and tensor.shape[1] == 3):
        raise ValueError(f"Input tensor must have shape (1, 3, H, W), but got {tensor.shape}")
    if tensor.dtype != torch.float32:
        print(f"Warning: Input tensor dtype is {tensor.dtype}, converting to float32.")
        tensor = tensor.float()

    # 2. Squeeze batch dimension, convert to NumPy, and permute dimensions
    # (1, 3, H, W) -> (3, H, W) [NumPy] -> (H, W, 3) [NumPy for OpenCV]
    image_np = tensor.squeeze(0).cpu().detach().numpy() # Shape: (3, H, W)
    image_np_hwc = np.transpose(image_np, (1, 2, 0))   # Shape: (H, W, 3)

    # 3. Scale float32 values [0.0, 1.0] to uint16 values [0, 65535]
    # Check min/max to confirm assumption about [0,1] range
    min_val, max_val = image_np_hwc.min(), image_np_hwc.max()
    # print(f"Tensor min/max before scaling: {min_val:.4f}, {max_val:.4f}")
    normalized_tensor = (image_np_hwc - min_val) / (max_val - min_val)
    if not (min_val >= -1e-5 and max_val <= 1.0 + 1e-5): # Allow for small float inaccuracies
        print(f"Warning: Tensor values are outside the expected [0, 1] range (min: {min_val:.2f}, max: {max_val:.2f}). "
              "Scaling might produce unexpected results. Values will be clamped to [0, 65535] after scaling by 65535.")
    
    # Perform scaling and conversion
    image_scaled = normalized_tensor* 65535.0
    # image_scaled = image_np_hwc * 65535.0
    image_16bit = np.clip(image_scaled, 0, 65535).astype(np.uint16)

    # 4. Save as 16-bit PNG using OpenCV
    # OpenCV's imwrite saves multi-channel (like RGB) arrays as they are.
    # If image_16bit is (H,W,3) in RGB order, it will be saved with RGB pixel values.
    # Standard image viewers might interpret 3-channel PNGs written by OpenCV as BGR
    # if they don't explicitly check PNG metadata for color space or if OpenCV itself
    # writes some BGR hint.
    # To ensure colors are displayed correctly in most viewers, explicitly convert RGB to BGR before saving.
    # However, for data integrity where you want to store the exact RGB values, save as is.
    
    # Option A: Save with RGB pixel values (data integrity focus)
    # image_to_save = image_16bit

    # Option B: Convert to BGR for better compatibility with some viewers (display focus)
    image_to_save = cv2.cvtColor(image_16bit, cv2.COLOR_RGB2BGR)
    # print("Note: Converting RGB to BGR before saving for better compatibility with some viewers.")


    try:
        success = cv2.imwrite(output_path, image_to_save, [cv2.IMWRITE_PNG_COMPRESSION, 0]) # No compression for speed
        # success = cv2.imwrite(output_path, image_to_save) # Default compression
        if success:
            # print(f"Successfully saved 16-bit RGB (as BGR for viewers) PNG to {output_path}")
            pass
        else:
            print(f"Failed to save image to {output_path} using OpenCV.")
    except Exception as e:
        print(f"Error saving image with OpenCV: {e}")

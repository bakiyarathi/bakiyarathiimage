import cv2
import numpy as np
from matplotlib import pyplot as plt


def remove_background(image_path):
    # Read the image with alpha channel
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if image has an alpha channel
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        rgb = img[:, :, :3]
    else:
        print("Image does not have an alpha channel.")
        return None

    # Create a mask
    mask = np.zeros(rgb.shape[:2], np.uint8)

    # Define the background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Define the rectangle around the foreground object
    height, width = rgb.shape[:2]
    rect = (10, 10, width - 10, height - 10)  # Adjust these values if necessary

    # Apply the GrabCut algorithm
    cv2.grabCut(rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the mask to the RGB image
    result_rgb = rgb * mask2[:, :, np.newaxis]

    # Combine with the original alpha channel
    result_alpha = alpha * mask2

    # Stack the RGB and alpha channels
    result = np.dstack((result_rgb, result_alpha))

    return result


# Example usage
image_path = 'C:\\Users\\Admin\\Downloads\\nameLogo.png'
result = remove_background(image_path)

if result is not None:
    # Save the result
    cv2.imwrite('result.png', result)

    # Display the result
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))
    plt.axis('off')
    plt.show()

# Basic imports
import numpy as np
import matplotlib.pyplot as plt

# Special imports for image preprocessing
import cv2
import PIL

# Read pickle files (vector shapes)
import pickle
import shapely


def open_pickle_polygons(pickle_path: str, read_option = '+rb') -> list:
    """Open a shapely object saved into a pickle file
       Transform the multipolygon object into a list of polygon objects"""
    
    with open(pickle_path, read_option) as f:
        polygons = pickle.load(f)
    
    return list(polygons.geoms)


def apply_mask_to_image(image: np.array, 
                        polygons: list, 
                        alpha: float, 
                        mask_color = (255, 255, 255), 
                        line_type = cv2.LINE_4) -> np.array:
    """Apply a list of polygonal masks to an image"""
    
    # Function to round coordinates of the exterior of the polygon
    int_coords = lambda x: np.array(x).round().astype(np.int32)

    # Make a copy we can modify
    image_to_mask = image.copy()

    # Iterate over the polygons
    for polygon in polygons:
        
        # Round the coordinates of polygon
        exterior = [int_coords(polygon.exterior.coords)]
        
        # Create an overlay image
        overlay = image_to_mask.copy()
        
        # Create the mask on the overlay
        cv2.fillPoly(overlay, exterior, color = mask_color, lineType = line_type)
        
        
        
        # Flatten the mask and the image layer
        cv2.addWeighted(overlay, alpha, image_to_mask, 1 - alpha, 0, image_to_mask)
    
    return image_to_mask
import numpy as np
import cv2
import yaml
from skimage.morphology import skeletonize
from CurvilinearTrack import CurvilinearTrack
from common import *

class PgmTrack(CurvilinearTrack):
    def __init__(self, pgm_path, yaml_path):
        super().__init__()

        # Load metadata from YAML
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        resolution = metadata['resolution']
        origin = np.array(metadata['origin'][:2])  # (x, y)

        # Load and preprocess PGM
        map_img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        map_img = cv2.medianBlur(map_img, 5)  # Remove noise
        
        # More flexible thresholding
        _, binary = cv2.threshold(map_img, 200, 255, cv2.THRESH_BINARY)

        # Invert: 0 = occupied, 1 = free
        free_space = (binary == 255).astype(np.uint8)

        # Skeletonize
        skeleton = skeletonize(free_space).astype(np.uint8)

        # Find skeleton points
        yx_coords = np.argwhere(skeleton == 1)
        if len(yx_coords) == 0:
            # Save debug images before failing
            cv2.imwrite('debug_threshold.png', binary)
            cv2.imwrite('debug_skeleton.png', skeleton*255)
            raise ValueError("No skeleton found - debug images saved")

        # Robust point chaining algorithm
        from scipy.spatial import cKDTree
        coords = yx_coords[:, [1, 0]]  # convert to (x, y)
        tree = cKDTree(coords)
        visited = [0]
        remaining_indices = set(range(len(coords))) - {0}

        while remaining_indices:
            # Find nearest unvisited point within reasonable distance
            dists, idxs = tree.query(coords[visited[-1]], k=min(20, len(coords)))
            for i in idxs:
                if i in remaining_indices:
                    visited.append(i)
                    remaining_indices.remove(i)
                    break
            else:  # No nearby points found
                if remaining_indices:
                    # Start new segment with remaining points
                    visited.append(next(iter(remaining_indices)))
                    remaining_indices.remove(visited[-1])

        sorted_coords = coords[visited]

        # Convert pixel to world coordinates
        world_coords = sorted_coords * resolution + origin

        # Build track
        self.buildContinuousTrack(world_coords)

# Demo
if __name__ == "__main__":
    pgm_path = "map_house.pgm"    # Updated to match your file
    yaml_path = "map_house.yaml"  # Updated to match your file

    track = PgmTrack(pgm_path, yaml_path)
    track.optimizePath(offset=0.15, visualize=True, save_gif=True)
    # track.save()
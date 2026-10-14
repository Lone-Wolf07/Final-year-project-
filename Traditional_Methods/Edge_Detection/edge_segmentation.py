import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology
from skimage.filters import threshold_isodata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import MIN_LUNG_FRACTION, clean_lungs, disk_radius, largest_component, run

RESULTS_DIR = Path(__file__).resolve().parent / "Results"
SIGMA_FRACTION = 0.004
LOW_QUANTILE = 0.80
HIGH_QUANTILE = 0.90
LINK_RADIUS_FRACTION = 0.008


def segment(img):
    sigma = max(1.0, SIGMA_FRACTION * min(img.shape))

    smoothed = filters.gaussian(img, sigma=sigma)
    gradient = filters.sobel(smoothed)

    edges = feature.canny(img, sigma=sigma, low_threshold=LOW_QUANTILE,
                          high_threshold=HIGH_QUANTILE, use_quantiles=True)

    link_radius = disk_radius(img, LINK_RADIUS_FRACTION)
    linked = ndi.binary_dilation(edges, structure=morphology.disk(link_radius))

    regions = measure.label(~linked, connectivity=1)
    props = measure.regionprops(regions, intensity_image=img)
    border = set(np.concatenate([regions[0], regions[-1], regions[:, 0], regions[:, -1]])) - {0}
    image_cut = threshold_isodata(img)
    outside = np.isin(regions, [r.label for r in props if r.label in border and r.intensity_mean < image_cut])
    body = ndi.binary_fill_holes(largest_component(~outside))

    body_cut = threshold_isodata(img[body]) if body.any() else image_cut
    min_area = MIN_LUNG_FRACTION * body.sum()
    lung_like = sorted(
        (r for r in props
         if r.label not in border and r.area >= min_area and r.intensity_mean < body_cut
         and body[tuple(r.coords[0])]),
        key=lambda r: r.area,
        reverse=True,
    )[:2]
    candidates = [regions == r.label for r in lung_like]

    grow = morphology.disk(link_radius)
    lungs = clean_lungs([ndi.binary_dilation(c, structure=grow) for c in candidates], body, img)

    panels = [
        ("Input", img),
        (f"1-2. Sobel gradient (sigma {sigma:.1f})", gradient),
        (f"3. Canny edges (q {LOW_QUANTILE}/{HIGH_QUANTILE})", edges),
        (f"4. Linked edges (dilated {link_radius} px)", linked),
        ("5. Body (inside the outer boundary)", body),
        ("6-7. Dark enclosed regions -> lungs", lungs),
    ]
    info = dict(sigma=round(sigma, 2), edge_fraction=round(edges.mean(), 4), region_cut=round(body_cut, 4))
    return lungs, body, panels, info


if __name__ == "__main__":
    run(segment, RESULTS_DIR, "Canny edge-based segmentation", "Edge detection")

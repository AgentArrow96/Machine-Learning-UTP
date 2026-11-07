"""Photos for the landing page (web/index.html), made from the original FloodNet test images.

Every photo is read exactly as the pipeline reads it (hash-checked, EXIF-oriented). Writes web/data/landing/*.webp
(committed). FloodNet-Supervised v1.0 is released under CDLA-Permissive-1.0; the page credits it.
"""
import numpy as np
from PIL import Image

from floodnet.common import config, environment, inside, log
from floodnet.data import Store, mask_array, remap
from floodnet.visuals import fit_long_side, oriented_photo

OUT = "web/data/landing"
PHOTOS = {
    "hero": ("test_7294", 1600),
    "roads": ("test_7611", 960),
    "homes": ("test_7423", 960),
    "authorities": ("test_7333", 960),
    "responders": ("test_7590", 960),
    "planners": ("test_7291", 960),
    "researchers": ("test_7464", 960),
    "river": ("test_7339", 1280),
    "street": ("test_7320", 1280),
    "edge": ("test_7457", 1280),
}
LABELLED = "test_7301"
MONTAGE = ["test_7457", "test_7333", "test_7461", "test_7315", "test_7300", "test_7415",
           "test_7413", "test_7323", "test_7577", "test_7583", "test_7241", "test_7486"]
COLOURS = {1: (245, 158, 11), 2: (8, 145, 178)}


def save(image, name):
    path = inside(f"{OUT}/{name}.webp")
    image.save(path, "WEBP", quality=80, method=6)
    return path.stat().st_size


def labelled(record, size):
    """Left half: the photo. Right half: the same photo with flooded buildings (amber) and roads (teal) painted in."""
    photo = oriented_photo(record)
    labels = remap(mask_array(inside(record["mask"])))
    if labels.shape != (photo.height, photo.width):
        raise SystemExit(f"Annotation and photo differ in size for {record['id']}")
    rgb = np.asarray(photo).astype(np.float32)
    painted = rgb.copy()
    for c, colour in COLOURS.items():
        painted[labels == c] = rgb[labels == c] * 0.35 + np.array(colour) * 0.65
    half = photo.width // 2
    rgb[:, half:] = painted[:, half:]
    rgb[:, half - 3:half + 3] = 255
    return fit_long_side(Image.fromarray(rgb.astype(np.uint8)), size)


def mask_layer(record, size):
    """Only the flooded buildings (amber) and flooded roads (teal), on a transparent layer."""
    labels = remap(mask_array(inside(record["mask"])))
    photo = oriented_photo(record)
    if labels.shape != (photo.height, photo.width):
        raise SystemExit(f"Annotation and photo differ in size for {record['id']}")
    rgba = np.zeros(labels.shape + (4,), np.uint8)
    for c, colour in COLOURS.items():
        rgba[labels == c] = (*colour, 235)
    layer = Image.fromarray(rgba)
    return layer.resize((size, round(size * layer.height / layer.width)), Image.Resampling.NEAREST)


def slic_layer(record, size):
    """SLIC superpixel outlines with the pipeline's settings (about 250 regions on a 256-px image), as a transparent layer."""
    from skimage.segmentation import find_boundaries, slic
    settings = config()["slic"]
    photo = oriented_photo(record)
    small = fit_long_side(photo, 256)
    labels = slic(np.asarray(small), n_segments=settings["n_segments"], compactness=settings["compactness"],
                  max_num_iter=settings["max_num_iter"], enforce_connectivity=True, start_label=1, channel_axis=-1)
    height = round(size * photo.height / photo.width)
    big = np.asarray(Image.fromarray(labels.astype(np.int32)).resize((size, height), Image.Resampling.NEAREST))
    edges = find_boundaries(big, mode="inner")
    rgba = np.zeros(big.shape + (4,), np.uint8)
    rgba[edges] = (125, 211, 252, 230)
    return Image.fromarray(rgba)


def montage(records, columns=4, tile=240):
    rows = -(-len(records) // columns)
    sheet = Image.new("RGB", (columns * tile, rows * tile * 3 // 4), (15, 23, 42))
    for k, record in enumerate(records):
        photo = fit_long_side(oriented_photo(record), tile).resize((tile - 6, tile * 3 // 4 - 6))
        sheet.paste(photo, ((k % columns) * tile + 3, (k // columns) * (tile * 3 // 4) + 3))
    return sheet


def main():
    environment()
    store = Store(config())
    inside(OUT).mkdir(parents=True, exist_ok=True)
    total = 0
    for name, (image_id, size) in PHOTOS.items():
        total += save(fit_long_side(oriented_photo(store.by_id[image_id]), size), name)
    total += save(labelled(store.by_id[LABELLED], 960), "labelled")
    total += save(montage([store.by_id[i] for i in MONTAGE]), "many")
    hero = store.by_id[PHOTOS["hero"][0]]
    total += save(fit_long_side(oriented_photo(hero), 960), "stack-photo")
    total += save(slic_layer(hero, 960), "stack-slic")
    total += save(mask_layer(hero, 960), "stack-mask")
    needle = store.by_id[LABELLED]
    total += save(fit_long_side(oriented_photo(needle), 960), "needle-photo")
    total += save(mask_layer(needle, 960), "needle-mask")
    log(f"Landing photos: {len(PHOTOS) + 7} files, {total / 2 ** 20:.1f} MiB in {OUT}")


if __name__ == "__main__":
    main()

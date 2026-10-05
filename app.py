"""
Edge Detection Explorer (Streamlit)
-----------------------------------
Upload an image -> get Sobel gradients, Non-Maximum Suppression,
double-threshold + hysteresis edges, gradient-direction colours,
Difference-of-Gaussian features, a Canny comparison and a detailed
statistical report about the edges found in the image.

Run locally:   streamlit run app.py
"""

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T.copy()  # [[-1,-2,-1],[0,0,0],[1,2,1]]


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------
def load_image(file, max_side: int):
    """Read an uploaded file -> (rgb uint8, gray uint8). Fixes EXIF rotation and
    shrinks large images so processing stays fast."""
    img = Image.open(file)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    rgb = np.asarray(img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray


def to_u8(a: np.ndarray, percentile: float = 100.0) -> np.ndarray:
    """Scale any float array to 0-255 uint8 for display."""
    a = np.abs(a).astype(np.float32)
    top = np.percentile(a, percentile) if percentile < 100 else a.max()
    if top <= 0:
        return np.zeros(a.shape, np.uint8)
    return np.clip(a / top * 255, 0, 255).astype(np.uint8)


def png_bytes(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    return buf.tobytes()


# --------------------------------------------------------------------------
# Edge detection pipeline (fully vectorised - no Python pixel loops)
# --------------------------------------------------------------------------
def sobel_gradients(gray: np.ndarray):
    """Return gradient magnitude (0-255 float) and gradient direction (0-180 deg)."""
    g = gray.astype(np.float32)
    gx = cv2.filter2D(g, -1, SOBEL_X, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(g, -1, SOBEL_Y, borderType=cv2.BORDER_REPLICATE)
    mag = np.hypot(gx, gy)
    if mag.max() > 0:
        mag = mag / mag.max() * 255.0
    theta = np.degrees(np.arctan2(gy, gx)) % 180.0
    return mag, theta


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """out[i, j] = a[i+dy, j+dx] (zero outside the image)."""
    h, w = a.shape
    p = np.pad(a, 1, mode="constant")
    return p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]


def non_max_suppression(mag: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Thin edges: keep a pixel only if it is the local maximum along the gradient."""
    q = np.round(theta / 45.0).astype(int) % 4  # 0:0deg 1:45deg 2:90deg 3:135deg
    neighbours = {
        0: ((0, -1), (0, 1)),     # gradient left/right
        1: ((1, 1), (-1, -1)),    # gradient down-right / up-left
        2: ((-1, 0), (1, 0)),     # gradient up/down
        3: ((1, -1), (-1, 1)),    # gradient down-left / up-right
    }
    keep = np.zeros(mag.shape, dtype=bool)
    for k, (d1, d2) in neighbours.items():
        a, b = _shift(mag, *d1), _shift(mag, *d2)
        keep |= (q == k) & (mag >= a) & (mag >= b)
    return np.where(keep, mag, 0.0)


def hysteresis(nms: np.ndarray, low: float, high: float):
    """Double threshold + edge tracking: weak edges survive only if connected to a strong one."""
    strong = nms >= high
    weak = nms >= low
    _, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    good = np.unique(labels[strong])
    good = good[good > 0]
    edges = np.isin(labels, good)
    return edges, strong


def direction_colors(theta: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Colour edge pixels by gradient direction (OpenCV hue range is 0-179 == 0-180 deg)."""
    hsv = np.zeros((*theta.shape, 3), np.uint8)
    hsv[..., 0] = np.clip(theta, 0, 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.where(edges, 255, 0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def difference_of_gaussian(gray: np.ndarray, levels: int, sigma0: float, k: float):
    """DoG(i) = G(sigma_i) - G(sigma_i+1). Works in float to avoid uint8 wrap-around."""
    g = gray.astype(np.float32)
    blurs, sigma = [], sigma0
    for _ in range(levels + 1):
        blurs.append(cv2.GaussianBlur(g, (0, 0), sigma))
        sigma *= k
    out = []
    for i in range(levels):
        out.append((to_u8(blurs[i] - blurs[i + 1], 99.5), sigma0 * k ** i))
    return out


@st.cache_data(show_spinner=False)
def run_pipeline(gray: np.ndarray, ksize: int, sigma: float,
                 low_pct: float, high_pct: float, canny_low: int, canny_high: int):
    blurred = gray if ksize <= 1 else cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    mag, theta = sobel_gradients(blurred)
    nms = non_max_suppression(mag, theta)
    edges, strong = hysteresis(nms, low_pct / 100 * 255, high_pct / 100 * 255)
    canny = cv2.Canny(blurred, canny_low, canny_high)
    return dict(blurred=blurred, mag=mag, theta=theta, nms=nms,
                edges=edges, strong=strong, canny=canny)


# --------------------------------------------------------------------------
# Analysis / report
# --------------------------------------------------------------------------
def analyse(res: dict, gray: np.ndarray) -> dict:
    edges, mag, theta = res["edges"], res["mag"], res["theta"]
    h, w = edges.shape
    n_edge = int(edges.sum())
    density = n_edge / (h * w) * 100
    n_segments = max(cv2.connectedComponents(edges.astype(np.uint8), connectivity=8)[0] - 1, 0)

    stats = dict(
        size=(w, h), edge_pixels=n_edge, density=density, segments=n_segments,
        mean_strength=float(mag[edges].mean()) if n_edge else 0.0,
        max_strength=float(mag[edges].max()) if n_edge else 0.0,
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        contrast=float(gray.std()),
    )

    # Orientation of the *edge line* = gradient direction + 90 degrees
    if n_edge:
        e = (theta[edges] + 90) % 180
        bins = {
            "Horizontal  ( — )": ((e < 22.5) | (e >= 157.5)),
            "Diagonal    ( \\ )": ((e >= 22.5) & (e < 67.5)),
            "Vertical    ( | )": ((e >= 67.5) & (e < 112.5)),
            "Diagonal    ( / )": ((e >= 112.5) & (e < 157.5)),
        }
        stats["orientation"] = {k: float(v.mean() * 100) for k, v in bins.items()}
        hist, _ = np.histogram(e, bins=18, range=(0, 180))
        stats["hist"] = hist
        # Average segment length (pixels per connected edge)
        stats["avg_segment"] = n_edge / max(n_segments, 1)
    else:
        stats["orientation"], stats["hist"], stats["avg_segment"] = {}, np.zeros(18), 0.0

    # 3x3 spatial grid: where are the edges?
    rows = ["Top", "Middle", "Bottom"]
    cols = ["Left", "Center", "Right"]
    grid = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            cell = edges[i * h // 3:(i + 1) * h // 3, j * w // 3:(j + 1) * w // 3]
            grid[i, j] = cell.mean() * 100 if cell.size else 0
    stats["grid"] = pd.DataFrame(grid, index=rows, columns=cols)
    return stats


def describe(stats: dict) -> list[str]:
    """Plain-language summary of what the numbers mean."""
    d = stats["density"]
    if d < 2:
        lvl = "very **smooth / minimal** (few edges, large flat regions)"
    elif d < 6:
        lvl = "**moderately detailed** (clear objects, some texture)"
    elif d < 12:
        lvl = "**highly detailed** (many objects or textured surfaces)"
    else:
        lvl = "**very busy / heavily textured** (edges almost everywhere, possibly noise)"

    s = stats["sharpness"]
    if s < 50:
        sharp = "**blurry / low focus**"
    elif s < 300:
        sharp = "**reasonably sharp**"
    else:
        sharp = "**very sharp or noisy**"

    lines = [
        f"The image is {lvl}: **{stats['density']:.2f}%** of pixels are edge pixels.",
        f"Image focus looks {sharp} (Laplacian variance = {s:.0f}).",
        f"**{stats['segments']:,}** separate edge segments were found, "
        f"averaging **{stats['avg_segment']:.0f}** pixels each "
        "(long segments = clear outlines, many tiny ones = texture/noise).",
    ]
    if stats["orientation"]:
        top = max(stats["orientation"], key=stats["orientation"].get)
        lines.append(f"Dominant edge direction: **{' '.join(top.split())}** "
                     f"({stats['orientation'][top]:.1f}% of edges).")
    g = stats["grid"]
    r, c = np.unravel_index(np.argmax(g.values), g.shape)
    lines.append(f"Most edges are in the **{g.index[r].lower()}-{g.columns[c].lower()}** "
                 f"region ({g.values[r, c]:.1f}% edge density there).")
    return lines


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Edge Detection Explorer", page_icon="🔍", layout="wide")
    st.title("🔍 Edge Detection Explorer")
    st.caption("Sobel • Non-Max Suppression • Hysteresis • Gradient Direction • "
               "Difference of Gaussian • Canny comparison")

    with st.sidebar:
        st.header("Settings")
        max_side = st.slider("Max image size (px)", 300, 1600, 800, 50,
                             help="Larger images are shrunk to this size for speed.")
        st.subheader("Pre-processing")
        ksize = st.select_slider("Gaussian blur kernel", [1, 3, 5, 7, 9], value=3,
                                 help="1 = no blur. More blur = less noise, fewer fine edges.")
        sigma = st.slider("Blur sigma (0 = auto)", 0.0, 5.0, 0.0, 0.1)
        st.subheader("Hysteresis thresholds (% of max gradient)")
        low_pct = st.slider("Low threshold", 1, 99, 10)
        high_pct = st.slider("High threshold", 1, 99, 25)
        if low_pct > high_pct:
            st.warning("Low threshold is above High - swapping them.")
            low_pct, high_pct = high_pct, low_pct
        st.subheader("Difference of Gaussian")
        dog_levels = st.slider("Levels", 2, 8, 6)
        dog_sigma = st.slider("Start sigma", 0.5, 3.0, 1.0, 0.1)
        st.subheader("Canny (comparison)")
        c_low = st.slider("Canny low", 0, 255, 50)
        c_high = st.slider("Canny high", 0, 255, 150)

    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"])
    if file is None:
        st.info("⬆️ Upload an image to start. Try a photo with clear objects (fruit, a building, a face).")
        return

    try:
        rgb, gray = load_image(file, max_side)
    except Exception as exc:
        st.error(f"Could not read this image: {exc}")
        return

    with st.spinner("Detecting edges..."):
        res = run_pipeline(gray, ksize, sigma, low_pct, high_pct, int(c_low), int(c_high))
        stats = analyse(res, gray)

    edge_img = (res["edges"] * 255).astype(np.uint8)

    tab_over, tab_steps, tab_dir, tab_dog, tab_cmp = st.tabs(
        ["📊 Overview & Report", "🪜 Step by step", "🧭 Gradient direction",
         "🌫️ Difference of Gaussian", "⚖️ Compare with Canny"])

    # ---- Overview ---------------------------------------------------------
    with tab_over:
        c1, c2 = st.columns(2)
        c1.image(rgb, caption="Original", use_container_width=True)
        c2.image(edge_img, caption="Detected edges (Sobel + NMS + Hysteresis)", use_container_width=True)
        st.download_button("⬇️ Download edge image (PNG)", png_bytes(edge_img),
                           "edges.png", "image/png")

        st.subheader("Edge report")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Image size", f"{stats['size'][0]} × {stats['size'][1]}")
        m2.metric("Edge pixels", f"{stats['edge_pixels']:,}")
        m3.metric("Edge density", f"{stats['density']:.2f}%")
        m4.metric("Edge segments", f"{stats['segments']:,}")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Avg edge strength", f"{stats['mean_strength']:.1f} / 255")
        m6.metric("Strongest edge", f"{stats['max_strength']:.1f} / 255")
        m7.metric("Sharpness", f"{stats['sharpness']:.0f}")
        m8.metric("Contrast (std)", f"{stats['contrast']:.1f}")

        st.markdown("#### What this means")
        for line in describe(stats):
            st.markdown(f"- {line}")

        if stats["orientation"]:
            left, right = st.columns(2)
            with left:
                st.markdown("#### Edge direction breakdown")
                st.dataframe(pd.DataFrame({"Share of edges (%)": stats["orientation"]}).round(1),
                             use_container_width=True)
                labels = [f"{i * 10}-{i * 10 + 10}°" for i in range(18)]
                st.bar_chart(pd.DataFrame({"pixels": stats["hist"]}, index=labels))
            with right:
                st.markdown("#### Where are the edges? (edge density %)")
                st.dataframe(stats["grid"].round(1).style.background_gradient(cmap="Blues"),
                             use_container_width=True)
        else:
            st.warning("No edges found. Lower the thresholds or reduce the blur in the sidebar.")

    # ---- Step by step -----------------------------------------------------
    with tab_steps:
        st.markdown("Each stage of the pipeline:")
        a, b = st.columns(2)
        a.image(gray, caption="1. Grayscale", use_container_width=True)
        b.image(res["blurred"], caption=f"2. Gaussian blur (kernel {ksize})", use_container_width=True)
        a.image(to_u8(res["mag"]), caption="3. Sobel gradient magnitude", use_container_width=True)
        b.image(to_u8(res["nms"]), caption="4. Non-Maximum Suppression (thin edges)", use_container_width=True)
        a.image((res["strong"] * 255).astype(np.uint8),
                caption="5a. Strong edges only (above high threshold)", use_container_width=True)
        b.image(edge_img, caption="5b. Final edges after hysteresis", use_container_width=True)
        with st.expander("How does it work?"):
            st.markdown(
                "1. **Sobel** slides two 3×3 kernels (horizontal & vertical) over the image; a large "
                "response means a big brightness change = an edge.\n"
                "2. **Magnitude** = √(Gx² + Gy²) tells *how strong* the edge is; the angle tells *which way* it faces.\n"
                "3. **NMS** keeps only the strongest pixel across the edge so lines become 1 pixel thin.\n"
                "4. **Double threshold + hysteresis** keeps strong edges and weak edges *only if* they touch a strong one.")

    # ---- Direction --------------------------------------------------------
    with tab_dir:
        a, b = st.columns([3, 1])
        a.image(direction_colors(res["theta"], res["edges"]),
                caption="Edge colour = gradient direction", use_container_width=True)
        wheel = np.zeros((180, 20, 3), np.uint8)
        wheel[..., 0] = np.arange(180, dtype=np.uint8)[:, None]
        wheel[..., 1:] = 255
        b.image(cv2.cvtColor(wheel, cv2.COLOR_HSV2RGB), caption="Colour scale 0° → 180° (top → bottom)",
                width=60)
        st.caption("Same colour = edges facing the same direction (e.g. all vertical lines share one colour).")

    # ---- DoG --------------------------------------------------------------
    with tab_dog:
        st.markdown("Subtracting a more-blurred image from a less-blurred one keeps only features "
                    "of a particular size. Higher levels highlight larger structures.")
        dogs = difference_of_gaussian(gray, dog_levels, dog_sigma, 1.6)
        cols = st.columns(3)
        for i, (im, s) in enumerate(dogs):
            cols[i % 3].image(im, caption=f"Level {i + 1}  (σ ≈ {s:.1f})", use_container_width=True)

    # ---- Canny ------------------------------------------------------------
    with tab_cmp:
        a, b = st.columns(2)
        a.image(edge_img, caption="Our Sobel + NMS + Hysteresis", use_container_width=True)
        b.image(res["canny"], caption="OpenCV Canny", use_container_width=True)
        ours, canny = res["edges"], res["canny"] > 0
        union = (ours | canny).sum()
        agree = (ours & canny).sum() / union * 100 if union else 0
        st.metric("Edge overlap (IoU)", f"{agree:.1f}%")
        st.caption(f"Ours: {ours.sum():,} px  |  Canny: {canny.sum():,} px. "
                   "Exact overlap is naturally low because edges are only 1 px thin; "
                   "visually they should look similar.")


if __name__ == "__main__":
    main()

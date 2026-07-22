import gradio as gr
import numpy as np
import cv2
import threading
import time
import os
from datetime import datetime

try:
    from arena_api.system import system
    ARENA_AVAILABLE = True
except ImportError:
    system = None
    ARENA_AVAILABLE = False

try:
    from lizad_client import LiZADClient
    LIZAD_AVAILABLE = True
except ImportError:
    LiZADClient = None
    LIZAD_AVAILABLE = False


def overlay_heatmap(frame_rgb, anomaly_map, alpha=0.45):
    h, w = frame_rgb.shape[:2]
    resized = cv2.resize(anomaly_map, (w, h))
    normalized = np.clip(resized * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(frame_rgb, 1 - alpha, heatmap_rgb, alpha, 0)

DATA_ROOT = "inspection_data/images"

# Hierarchical label taxonomy: category -> list of subtypes.
# "normal" is a leaf category (images go straight in inspection_data/images/normal/).
# "defect" is a parent whose subtypes are the actual capture folders
# (inspection_data/images/defect/missing_component/, etc.) - a category with
# subtypes never gets images saved directly under it.
TAXONOMY = {
    "normal": [],
    "defect": ["missing_component", "wrong_placement", "wrong_color"],
}

# Tesla UI Design Standards palette (confluence.teslamotors.com/spaces/CONHUB/pages/4179687050)
COLOR_PRIMARY_BLUE = "#3e6be2"
COLOR_WHITE = "#FFFFFF"
COLOR_GREY_BG = "#F4F4F4"
COLOR_STATUS_RED = "#eb432f"
COLOR_STATUS_ORANGE = "#f29137"
COLOR_STATUS_GREEN = "#02b028"
COLOR_STATUS_TEAL = "#3eade1"
COLOR_STATUS_GREY = "#707070"


class CameraController:
    def __init__(self):
        self.device = None
        self.latest_frame = None
        self.running = False
        self.connected = False
        self.lock = threading.Lock()

    def connect(self):
        if not ARENA_AVAILABLE:
            return False
        devices = system.create_device()
        if not devices:
            return False
        self.device = devices[0]
        nodemap = self.device.nodemap
        nodemap.get_node("PixelFormat").value = "BGR8"
        nodemap.get_node("ExposureAuto").value = "Continuous"
        nodemap.get_node("GainAuto").value = "Continuous"
        try:
            nodemap.get_node("BalanceWhiteAuto").value = "Continuous"
        except Exception:
            pass  # not every camera/firmware exposes this node
        self.device.start_stream()
        self.running = True
        self.connected = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        return True

    def _capture_loop(self):
        while self.running:
            try:
                buffer = self.device.get_buffer()
                image = np.ctypeslib.as_array(
                    buffer.pdata, shape=(buffer.height, buffer.width, 3)
                ).copy()
                self.device.requeue_buffer(buffer)
                with self.lock:
                    self.latest_frame = image
            except Exception as e:
                print(f"Capture error: {e}")
                self.connected = False
                time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)

    def set_exposure_auto(self, auto):
        self.device.nodemap.get_node("ExposureAuto").value = "Continuous" if auto else "Off"

    def set_exposure_value(self, value):
        nodemap = self.device.nodemap
        nodemap.get_node("ExposureAuto").value = "Off"
        nodemap.get_node("ExposureTime").value = float(value)

    def set_gain_auto(self, auto):
        self.device.nodemap.get_node("GainAuto").value = "Continuous" if auto else "Off"

    def set_gain_value(self, value):
        nodemap = self.device.nodemap
        nodemap.get_node("GainAuto").value = "Off"
        nodemap.get_node("Gain").value = float(value)

    def set_wb_auto(self, auto):
        try:
            self.device.nodemap.get_node("BalanceWhiteAuto").value = "Continuous" if auto else "Off"
        except Exception:
            pass

    def get_current_exposure(self):
        if not self.device:
            return None
        return float(self.device.nodemap.get_node("ExposureTime").value)

    def get_current_gain(self):
        if not self.device:
            return None
        return float(self.device.nodemap.get_node("Gain").value)

    def stop(self):
        self.running = False
        if self.device:
            self.device.stop_stream()
            system.destroy_device(self.device)


camera = CameraController()
last_saved_path = {"path": None}

inference_state = {
    "engine": None,
    "enabled": False,
    "threshold": 0.5,
    "latest_overlay": None,
    "latest_score": 0.0,
}


def toggle_inference(enabled):
    if enabled and not LIZAD_AVAILABLE:
        return "LiZAD client not available on this machine", gr.update(value=False)
    inference_state["enabled"] = enabled
    if enabled and inference_state["engine"] is None:
        client = LiZADClient(host="localhost", port=8000)
        if not client.health():
            inference_state["enabled"] = False
            return "Cannot reach LiZAD inference server on localhost:8000 - is lizad_server.py running?", gr.update(value=False)
        inference_state["engine"] = client
    return ("Live inference enabled" if enabled else "Live inference disabled"), gr.update()


def update_threshold(value):
    inference_state["threshold"] = value


def all_images_in_folder(folder):
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder)]


def score_images(engine, paths):
    scores = []
    for path in paths:
        frame = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        _, score = engine.run(frame)
        scores.append(score)
    return scores


def suggest_threshold():
    engine = inference_state["engine"]
    if engine is None:
        return gr.update(), "Enable Live Inference first so the LiZAD connection is available."

    tree = discover_taxonomy()
    normal_paths = all_images_in_folder(label_dir("normal", None))
    defect_paths = []
    for subtype in tree.get("defect", []):
        defect_paths.extend(all_images_in_folder(label_dir("defect", subtype)))

    if not normal_paths or not defect_paths:
        return gr.update(), (
            f"Need at least one captured image in both 'normal' and a 'defect' subtype "
            f"to suggest a threshold (have {len(normal_paths)} normal, {len(defect_paths)} defect)."
        )

    normal_scores = score_images(engine, normal_paths)
    defect_scores = score_images(engine, defect_paths)
    max_normal = max(normal_scores)
    min_defect = min(defect_scores)
    suggested = round((max_normal + min_defect) / 2, 3)

    if max_normal >= min_defect:
        note = (
            f"Scores overlap (normal max={max_normal:.3f}, defect min={min_defect:.3f}) - "
            "this is a rough midpoint, expect some misses. Capture more examples or revisit the prompts."
        )
    else:
        note = f"Clean separation - normal max={max_normal:.3f}, defect min={min_defect:.3f}."

    return (
        gr.update(value=suggested),
        f"Suggested threshold: {suggested}. {note} ({len(normal_scores)} normal, {len(defect_scores)} defect images evaluated)",
    )


def lock_threshold(locked):
    return gr.update(interactive=not locked), gr.update(interactive=not locked)


def _inference_loop():
    while True:
        if inference_state["enabled"] and inference_state["engine"] is not None:
            frame = camera.get_frame()
            if frame is not None:
                try:
                    anomaly_map, score = inference_state["engine"].run(frame)
                    inference_state["latest_overlay"] = overlay_heatmap(frame, anomaly_map)
                    inference_state["latest_score"] = score
                except Exception as e:
                    print(f"Inference error: {e}")
        time.sleep(1.0)


threading.Thread(target=_inference_loop, daemon=True).start()


def verdict_html(score, threshold):
    if score >= threshold:
        return f'<div style="font-size:2em;font-weight:700;color:{COLOR_STATUS_RED};">⚠ ANOMALY DETECTED</div>'
    return f'<div style="font-size:2em;font-weight:700;color:{COLOR_STATUS_GREEN};">✓ NORMAL</div>'


def stream_inference():
    while True:
        overlay = inference_state["latest_overlay"]
        score = inference_state["latest_score"]
        threshold = inference_state["threshold"]
        if overlay is not None:
            yield overlay, round(score, 3), verdict_html(score, threshold)
        else:
            yield no_camera_placeholder(), 0.0, '<div style="font-size:1.2em;color:#707070;">Waiting for inference...</div>'
        time.sleep(0.3)


def status_html():
    if camera.connected:
        return f'<div style="display:flex;align-items:center;gap:8px;font-weight:600;"><span style="width:14px;height:14px;border-radius:50%;background:{COLOR_STATUS_GREEN};display:inline-block;"></span>Camera Connected</div>'
    return f'<div style="display:flex;align-items:center;gap:8px;font-weight:600;"><span style="width:14px;height:14px;border-radius:50%;background:{COLOR_STATUS_RED};display:inline-block;"></span>No Camera</div>'


def compute_sharpness(frame):
    if frame is None:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)


def compute_brightness(frame):
    if frame is None:
        return 0.0
    return round(float(np.mean(frame)), 1)


def no_camera_placeholder():
    img = np.full((480, 640, 3), 240, dtype=np.uint8)
    cv2.putText(img, "No Camera Connected", (70, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (112, 112, 112), 2, cv2.LINE_AA)
    cv2.putText(img, "Waiting for device...", (70, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (112, 112, 112), 1, cv2.LINE_AA)
    return img


def stream_frames():
    while True:
        frame = camera.get_frame()
        if frame is not None:
            sharpness = compute_sharpness(frame)
            brightness = compute_brightness(frame)
            yield frame, frame, sharpness, sharpness, brightness, status_html()
        else:
            placeholder = no_camera_placeholder()
            yield placeholder, placeholder, 0.0, 0.0, 0.0, status_html()
        time.sleep(0.2)


def toggle_exposure_auto(auto):
    camera.set_exposure_auto(auto)
    return gr.update(interactive=not auto)


def update_exposure(value):
    camera.set_exposure_value(value)


def toggle_gain_auto(auto):
    camera.set_gain_auto(auto)
    return gr.update(interactive=not auto)


def update_gain(value):
    camera.set_gain_value(value)


def toggle_wb_auto(auto):
    camera.set_wb_auto(auto)


def lock_camera_settings(locked, exposure_auto_value, gain_auto_value):
    if locked:
        exposure_val = camera.get_current_exposure()
        gain_val = camera.get_current_gain()
        if exposure_val is None or gain_val is None:
            return (
                gr.update(value=False),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                "Cannot lock - no camera connected",
            )
        camera.set_exposure_value(exposure_val)
        camera.set_gain_value(gain_val)
        return (
            gr.update(),
            gr.update(interactive=False),
            gr.update(value=exposure_val, interactive=False),
            gr.update(interactive=False),
            gr.update(value=gain_val, interactive=False),
            gr.update(interactive=False),
            f"Locked - exposure {exposure_val:.0f}µs, gain {gain_val:.1f}dB frozen",
        )
    return (
        gr.update(),
        gr.update(interactive=True),
        gr.update(interactive=not exposure_auto_value),
        gr.update(interactive=True),
        gr.update(interactive=not gain_auto_value),
        gr.update(interactive=True),
        "Unlocked - settings can be adjusted",
    )


def label_from_choice(choice):
    return choice.split(" (")[0] if choice else None


def label_dir(category, subtype):
    return os.path.join(DATA_ROOT, category, subtype) if subtype else os.path.join(DATA_ROOT, category)


def count_images(category, subtype):
    folder = label_dir(category, subtype)
    return len(os.listdir(folder)) if os.path.isdir(folder) else 0


def discover_taxonomy():
    """Merge the built-in TAXONOMY with whatever category/subtype folders
    already exist on disk, so subtypes added via the custom-subtype box in
    past sessions still show up."""
    tree = {category: list(subs) for category, subs in TAXONOMY.items()}
    if not os.path.isdir(DATA_ROOT):
        return tree
    for category in os.listdir(DATA_ROOT):
        if not os.path.isdir(os.path.join(DATA_ROOT, category)):
            continue
        tree.setdefault(category, [])
        cat_path = os.path.join(DATA_ROOT, category)
        for sub in os.listdir(cat_path):
            if os.path.isdir(os.path.join(cat_path, sub)) and sub not in tree[category]:
                tree[category].append(sub)
    return tree


def category_choices(tree):
    choices = []
    for category in sorted(tree):
        subs = tree[category]
        total = sum(count_images(category, s) for s in subs) if subs else count_images(category, None)
        choices.append(f"{category} ({total})")
    return choices


def subtype_choices(tree, category):
    return [f"{s} ({count_images(category, s)})" for s in sorted(tree.get(category, []))]


def choice_for_value(value, choices):
    if not value:
        return None
    return next((c for c in choices if c.startswith(value + " (")), (choices[0] if choices else None))


def current_selection(category_choice, subtype_choice):
    category = label_from_choice(category_choice)
    subtype = label_from_choice(subtype_choice) if subtype_choice else None
    return category, subtype


def images_for(category, subtype):
    folder = label_dir(category, subtype)
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder))[-8:]


def on_category_change(category_choice):
    tree = discover_taxonomy()
    category = label_from_choice(category_choice)
    subs = subtype_choices(tree, category)
    sub_value = subs[0] if subs else None
    subtype = label_from_choice(sub_value) if sub_value else None
    return gr.update(choices=subs, value=sub_value), images_for(category, subtype), None, ""


def on_subtype_change(category_choice, subtype_choice):
    category, subtype = current_selection(category_choice, subtype_choice)
    return images_for(category, subtype), None, ""


def on_gallery_select(evt: gr.SelectData, category_choice, subtype_choice):
    category, subtype = current_selection(category_choice, subtype_choice)
    images = images_for(category, subtype)
    if evt.index < len(images):
        path = images[evt.index]
        return path, os.path.basename(path)
    return None, ""


def delete_selected_image(selected_path, category_choice, subtype_choice):
    category, subtype = current_selection(category_choice, subtype_choice)
    if selected_path and os.path.exists(selected_path):
        os.remove(selected_path)
        msg = f"Deleted {selected_path}"
    else:
        msg = "Nothing selected to delete"

    tree = discover_taxonomy()
    new_categories = category_choices(tree)
    new_category_value = choice_for_value(category, new_categories)
    new_subtypes = subtype_choices(tree, category)
    new_subtype_value = choice_for_value(subtype, new_subtypes)

    return (
        msg,
        gr.update(choices=new_categories, value=new_category_value),
        gr.update(choices=new_subtypes, value=new_subtype_value),
        images_for(category, subtype),
        None,
        "",
    )


def capture_and_save(category_choice, subtype_choice, custom_subtype):
    frame = camera.get_frame()
    if frame is None:
        tree = discover_taxonomy()
        category = label_from_choice(category_choice)
        return (
            "No frame available yet - is the camera connected?",
            gr.update(choices=category_choices(tree)),
            gr.update(choices=subtype_choices(tree, category)),
            [],
            "",
        )

    category, subtype = current_selection(category_choice, subtype_choice)
    if custom_subtype.strip():
        subtype = custom_subtype.strip()

    folder = label_dir(category, subtype)
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath = os.path.join(folder, f"{timestamp}.jpg")
    cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    last_saved_path["path"] = filepath

    tree = discover_taxonomy()
    new_categories = category_choices(tree)
    new_category_value = choice_for_value(category, new_categories)
    new_subtypes = subtype_choices(tree, category)
    new_subtype_value = choice_for_value(subtype, new_subtypes)

    return (
        f"Saved to {filepath}",
        gr.update(choices=new_categories, value=new_category_value),
        gr.update(choices=new_subtypes, value=new_subtype_value),
        images_for(category, subtype),
        "",
    )


def undo_last_capture(category_choice, subtype_choice):
    path = last_saved_path["path"]
    if path and os.path.exists(path):
        os.remove(path)
        last_saved_path["path"] = None
        msg = f"Removed {path}"
    else:
        msg = "Nothing to undo"

    category, subtype = current_selection(category_choice, subtype_choice)
    tree = discover_taxonomy()
    new_categories = category_choices(tree)
    new_category_value = choice_for_value(category, new_categories)
    new_subtypes = subtype_choices(tree, category)
    new_subtype_value = choice_for_value(subtype, new_subtypes)

    return (
        msg,
        gr.update(choices=new_categories, value=new_category_value),
        gr.update(choices=new_subtypes, value=new_subtype_value),
        images_for(category, subtype),
    )


# Light, high-contrast theme built from the Tesla UI Design Standards palette,
# not a generic "dark mode" look - this is a glance-readable HMI, not a dev tool demo.
THEME = gr.themes.Default(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
).set(
    body_background_fill=COLOR_GREY_BG,
    body_background_fill_dark=COLOR_GREY_BG,
    block_background_fill=COLOR_WHITE,
    block_background_fill_dark=COLOR_WHITE,
    block_border_color="#dddddd",
    button_primary_background_fill=COLOR_PRIMARY_BLUE,
    button_primary_background_fill_hover="#3457c2",
    button_primary_text_color=COLOR_WHITE,
    button_secondary_background_fill=COLOR_WHITE,
    button_secondary_border_color=COLOR_STATUS_GREY,
    button_secondary_text_color=COLOR_PRIMARY_BLUE,
)

CSS = """
.gradio-container { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }

#header {
    text-align: center;
    padding: 1.25em 0 1em 0;
    margin-bottom: 0.75em;
    border-bottom: 3px solid #3e6be2;
}
#header h1 {
    font-weight: 800;
    letter-spacing: -0.03em;
    font-size: 1.8em;
    color: #1a1a1a;
    margin: 0;
}
#header p {
    color: #707070;
    font-size: 0.95em;
    margin-top: 0.25em;
}

.card {
    border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.08) !important;
    border: 1px solid #e5e5e5 !important;
}

#live-feed-panel {
    border-radius: 14px !important;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(0,0,0,0.10) !important;
}

button.lg { border-radius: 10px !important; font-weight: 600 !important; }

#delete-btn { background: #eb432f !important; color: white !important; border: none !important; }
#delete-btn:hover { background: #c93520 !important; }

/* Make the capture gallery expand to fill remaining vertical space at the
   bottom of the tab, but cap individual thumbnail height by targeting the
   <img> tags directly rather than Gradio's internal wrapper class name
   (which changed between Gradio 5.x and 6.x and silently broke this rule -
   without a cap, a handful of images stretch to fill the whole container). */
.gradio-container { display: flex; flex-direction: column; min-height: 100vh; }
gradio-app, .tabs, .tabitem { display: flex; flex-direction: column; flex: 1; }
#capture-gallery {
    flex: 1 1 auto;
    min-height: 220px;
}
#capture-gallery img {
    max-height: 180px !important;
    object-fit: contain !important;
}
"""

with gr.Blocks(title="ArgusVision", theme=THEME, css=CSS) as demo:
    with gr.Column(elem_id="header"):
        gr.Markdown("# ArgusVision")
        gr.Markdown("Live capture and dataset-building tool for the LiZAD zero-shot anomaly detection prototype")

    with gr.Tabs():
        with gr.Tab("Live Capture"):
            status_display = gr.HTML(status_html())

            with gr.Row():
                with gr.Column(scale=3):
                    live_feed = gr.Image(label="Live Feed", streaming=True, height=560)
                    sharpness_display = gr.Number(label="Focus Sharpness (higher = sharper)", interactive=False)

                with gr.Column(scale=1):
                    initial_tree = discover_taxonomy()
                    initial_categories = category_choices(initial_tree)
                    category_radio = gr.Radio(choices=initial_categories, label="Category", value=initial_categories[0])
                    initial_category = label_from_choice(initial_categories[0])
                    initial_subtypes = subtype_choices(initial_tree, initial_category)
                    subtype_radio = gr.Radio(
                        choices=initial_subtypes,
                        label="Subtype (defect type, etc.)",
                        value=initial_subtypes[0] if initial_subtypes else None,
                    )
                    custom_subtype = gr.Textbox(label="Or type a new subtype", placeholder="e.g. bent_pin")
                    capture_btn = gr.Button("Capture", variant="primary", size="lg")
                    undo_btn = gr.Button("Undo Last", variant="secondary", size="lg")
                    status = gr.Textbox(label="Status", interactive=False)
                    gr.Markdown("Click a thumbnail below to select it, then delete it.")
                    selected_image_display = gr.Textbox(label="Selected Image", interactive=False)
                    delete_btn = gr.Button("Delete Selected", elem_id="delete-btn", size="lg")

            selected_image_path = gr.State(None)

            gallery = gr.Gallery(
                label="Recent Captures for This Category/Subtype",
                columns=6,
                rows=1,
                object_fit="contain",
                preview=False,
                elem_id="capture-gallery",
            )

        with gr.Tab("Inspection"):
            inference_toggle = gr.Checkbox(label="Enable Live Inference (LiZAD, trained_on_visa)", value=False)
            inference_status = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                with gr.Column(scale=3):
                    inference_overlay = gr.Image(label="Anomaly Heatmap", height=560)

                with gr.Column(scale=1):
                    threshold_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                        label="Anomaly Threshold"
                    )
                    suggest_threshold_btn = gr.Button(
                        "Suggest Threshold from Captured Images", variant="secondary"
                    )
                    threshold_suggestion_status = gr.Textbox(label="Suggestion Result", interactive=False)
                    lock_threshold_checkbox = gr.Checkbox(label="Lock Threshold", value=False)
                    score_display = gr.Number(label="Anomaly Score (max, 0-1)", interactive=False)
                    verdict_display = gr.HTML()

        with gr.Tab("Camera Settings"):
            gr.Markdown(
                "Let **Auto** settle on a good image using the preview below "
                "(aim for sharpness above ~50 and brightness in the 100-180 range), "
                "then check **Lock Camera Settings** to freeze exactly what's currently "
                "active so it can't drift or get bumped during production."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    camera_settings_preview = gr.Image(label="Live Preview", height=360)
                    with gr.Row():
                        cs_sharpness_display = gr.Number(label="Sharpness (higher = sharper)", interactive=False)
                        cs_brightness_display = gr.Number(label="Brightness (target 100-180)", interactive=False)

                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### Exposure — Manual / Auto")
                        exposure_auto = gr.Checkbox(label="Auto Exposure", value=True)
                        exposure_slider = gr.Slider(
                            minimum=10, maximum=100000, value=10000,
                            label="Exposure Time (microseconds)", interactive=False
                        )

                    with gr.Group():
                        gr.Markdown("### Gain — Manual / Auto")
                        gain_auto = gr.Checkbox(label="Auto Gain", value=True)
                        gain_slider = gr.Slider(
                            minimum=0, maximum=48, value=0,
                            label="Gain (dB)", interactive=False
                        )

                    with gr.Group():
                        gr.Markdown("### White Balance — Manual / Auto")
                        wb_auto = gr.Checkbox(label="Auto White Balance", value=True)

                    lock_camera_checkbox = gr.Checkbox(label="Lock Camera Settings", value=False)
                    lock_camera_status = gr.Textbox(label="Lock Status", interactive=False)

    exposure_auto.change(toggle_exposure_auto, inputs=exposure_auto, outputs=exposure_slider)
    exposure_slider.release(update_exposure, inputs=exposure_slider)

    gain_auto.change(toggle_gain_auto, inputs=gain_auto, outputs=gain_slider)
    gain_slider.release(update_gain, inputs=gain_slider)

    wb_auto.change(toggle_wb_auto, inputs=wb_auto)

    lock_camera_checkbox.change(
        lock_camera_settings,
        inputs=[lock_camera_checkbox, exposure_auto, gain_auto],
        outputs=[lock_camera_checkbox, exposure_auto, exposure_slider, gain_auto, gain_slider, wb_auto, lock_camera_status],
    )

    suggest_threshold_btn.click(
        suggest_threshold,
        outputs=[threshold_slider, threshold_suggestion_status],
    )
    lock_threshold_checkbox.change(
        lock_threshold,
        inputs=lock_threshold_checkbox,
        outputs=[threshold_slider, suggest_threshold_btn],
    )

    category_radio.change(
        on_category_change,
        inputs=category_radio,
        outputs=[subtype_radio, gallery, selected_image_path, selected_image_display],
    )
    subtype_radio.change(
        on_subtype_change,
        inputs=[category_radio, subtype_radio],
        outputs=[gallery, selected_image_path, selected_image_display],
    )

    capture_btn.click(
        capture_and_save,
        inputs=[category_radio, subtype_radio, custom_subtype],
        outputs=[status, category_radio, subtype_radio, gallery, custom_subtype],
    )
    undo_btn.click(
        undo_last_capture,
        inputs=[category_radio, subtype_radio],
        outputs=[status, category_radio, subtype_radio, gallery],
    )

    gallery.select(
        on_gallery_select,
        inputs=[category_radio, subtype_radio],
        outputs=[selected_image_path, selected_image_display],
    )
    delete_btn.click(
        delete_selected_image,
        inputs=[selected_image_path, category_radio, subtype_radio],
        outputs=[status, category_radio, subtype_radio, gallery, selected_image_path, selected_image_display],
    )

    inference_toggle.change(toggle_inference, inputs=inference_toggle, outputs=[inference_status, inference_toggle])
    threshold_slider.change(update_threshold, inputs=threshold_slider)

    demo.load(
        stream_frames,
        outputs=[live_feed, camera_settings_preview, sharpness_display, cs_sharpness_display, cs_brightness_display, status_display],
    )
    demo.load(stream_inference, outputs=[inference_overlay, score_display, verdict_display])


if __name__ == "__main__":
    if not camera.connect():
        print("WARNING: No camera found at startup - live feed will stay blank until one connects")
    demo.launch(server_name="0.0.0.0", server_port=7860)

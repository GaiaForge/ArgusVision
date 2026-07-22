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
PRESET_LABELS = ["normal", "missing_component", "wrong_placement", "wrong_color"]

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
        return "LiZAD engine not available on this machine", gr.update(value=False)
    inference_state["enabled"] = enabled
    if enabled and inference_state["engine"] is None:
        try:
            inference_state["engine"] = LiZADEngine(CHECKPOINT_PATH, class_name="pcb")
        except Exception as e:
            inference_state["enabled"] = False
            return f"Failed to load LiZAD engine: {e}", gr.update(value=False)
    return ("Live inference enabled" if enabled else "Live inference disabled"), gr.update()


def update_threshold(value):
    inference_state["threshold"] = value


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


def no_camera_placeholder():
    img = np.full((480, 640, 3), 240, dtype=np.uint8)
    cv2.putText(img, "No Camera Connected", (70, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (112, 112, 112), 2, cv2.LINE_AA)
    cv2.putText(img, "Waiting for device...", (70, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (112, 112, 112), 1, cv2.LINE_AA)
    return img


def stream_frames():
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield frame, compute_sharpness(frame), status_html()
        else:
            yield no_camera_placeholder(), 0.0, status_html()
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


def label_from_choice(choice):
    return choice.split(" (")[0] if choice else "normal"


def label_counts():
    labels = set(PRESET_LABELS)
    if os.path.isdir(DATA_ROOT):
        labels.update(
            d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))
        )
    choices = []
    for label in sorted(labels):
        folder = os.path.join(DATA_ROOT, label)
        count = len(os.listdir(folder)) if os.path.isdir(folder) else 0
        choices.append(f"{label} ({count})")
    return choices


def choice_for_label(label, choices):
    return next((c for c in choices if c.startswith(label + " (")), choices[0])


def images_for_label(label):
    folder = os.path.join(DATA_ROOT, label)
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder))[-8:]


def refresh_gallery(label_choice):
    return images_for_label(label_from_choice(label_choice))


def capture_and_save(label_choice, custom_label):
    frame = camera.get_frame()
    if frame is None:
        empty_choices = label_counts()
        return "No frame available yet - is the camera connected?", gr.update(choices=empty_choices), [], ""

    label = custom_label.strip() if custom_label.strip() else label_from_choice(label_choice)
    folder = os.path.join(DATA_ROOT, label)
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath = os.path.join(folder, f"{timestamp}.jpg")
    cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    last_saved_path["path"] = filepath

    new_choices = label_counts()
    new_value = choice_for_label(label, new_choices)

    return (
        f"Saved to {filepath}",
        gr.update(choices=new_choices, value=new_value),
        images_for_label(label),
        "",
    )


def undo_last_capture(label_choice):
    path = last_saved_path["path"]
    if path and os.path.exists(path):
        os.remove(path)
        last_saved_path["path"] = None
        msg = f"Removed {path}"
    else:
        msg = "Nothing to undo"

    current_label = label_from_choice(label_choice)
    new_choices = label_counts()
    new_value = choice_for_label(current_label, new_choices)
    return msg, gr.update(choices=new_choices, value=new_value), images_for_label(current_label)


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
                    initial_choices = label_counts()
                    label_radio = gr.Radio(choices=initial_choices, label="Label", value=initial_choices[0])
                    custom_label = gr.Textbox(label="Or type a new label", placeholder="e.g. bent_pin")
                    capture_btn = gr.Button("Capture", variant="primary", size="lg")
                    undo_btn = gr.Button("Undo Last", variant="secondary", size="lg")
                    status = gr.Textbox(label="Status", interactive=False)

            gallery = gr.Gallery(label="Recent Captures for This Label", columns=8, height=150)

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
                    score_display = gr.Number(label="Anomaly Score (max, 0-1)", interactive=False)
                    verdict_display = gr.HTML()

        with gr.Tab("Camera Settings"):
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

    exposure_auto.change(toggle_exposure_auto, inputs=exposure_auto, outputs=exposure_slider)
    exposure_slider.release(update_exposure, inputs=exposure_slider)

    gain_auto.change(toggle_gain_auto, inputs=gain_auto, outputs=gain_slider)
    gain_slider.release(update_gain, inputs=gain_slider)

    wb_auto.change(toggle_wb_auto, inputs=wb_auto)

    label_radio.change(refresh_gallery, inputs=label_radio, outputs=gallery)

    capture_btn.click(
        capture_and_save,
        inputs=[label_radio, custom_label],
        outputs=[status, label_radio, gallery, custom_label],
    )
    undo_btn.click(undo_last_capture, inputs=label_radio, outputs=[status, label_radio, gallery])

    inference_toggle.change(toggle_inference, inputs=inference_toggle, outputs=[inference_status, inference_toggle])
    threshold_slider.change(update_threshold, inputs=threshold_slider)

    demo.load(stream_frames, outputs=[live_feed, sharpness_display, status_display])
    demo.load(stream_inference, outputs=[inference_overlay, score_display, verdict_display])


if __name__ == "__main__":
    if not camera.connect():
        print("WARNING: No camera found at startup - live feed will stay blank until one connects")
    demo.launch(server_name="0.0.0.0", server_port=7860)

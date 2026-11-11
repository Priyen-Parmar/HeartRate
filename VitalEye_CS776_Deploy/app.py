import os
import sys

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

# If Gradio is not installed locally, automatically launch the live webcam HR monitor
try:
    import gradio as gr
except ImportError:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    print("ℹ Gradio is not installed in this local environment — launching live webcam HR monitor instead...")
    from live_hr_monitor import run, parse_args
    if __name__ == "__main__":
        run(parse_args())
    sys.exit(0)

import sliding_pipeline
from datetime import datetime
from types import SimpleNamespace

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
yolo_weights = os.path.join(DEPLOY_DIR, "models", "yolov8", "weights", "best.pt")
physformer_weights = os.path.join(DEPLOY_DIR, "models", "vidformer.pt")


def process_video(video_path):
    if video_path is None:
        raise gr.Error("Please upload a video file.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(DEPLOY_DIR, "outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    out_mp4 = os.path.join(output_dir, "annotated_output.mp4")

    args = SimpleNamespace(
        video=video_path,
        yolo_weights=yolo_weights,
        physformer_weights=physformer_weights,
        hf_repo="pushpender-23/yolo-cs776-model",
        yolo_hf_file="train_medium_v1/weights/best.pt",
        physformer_hf_file="physformer.pt",
        output=out_mp4,
        window_sec=10,
        stride_sec=1,
    )

    try:
        gr.Info("Initializing YOLO face tracking & native 3D-CNN continuous sequence...")
        final_bpm, final_rr, plot_path, annotated_video = sliding_pipeline.main(args)
    except Exception as e:
        raise gr.Error(f"Pipeline Execution Failed: {str(e)}")

    bpm_text = f"{final_bpm:.1f} BPM"
    rr_text = f"{final_rr:.1f} BRPM"
    gr.Info("Processing successfully completed!")

    return bpm_text, rr_text, plot_path, annotated_video


with gr.Blocks(title="PhysformerX rPPG Analytics") as demo:
    gr.Markdown("# 🫀 PhysformerX rPPG & YOLOv8 Analytics")
    gr.Markdown("Upload a video to perform causal body-part tracking and compute Heart Rate & Respiratory Rate continuously.")

    with gr.Row():
        with gr.Column():
            video_input = gr.File(
                label="Input Video (Accepts AVI, MP4, MKV, MOV...)",
                file_types=["video", ".avi", ".mkv", ".mov", ".mp4"],
                type="filepath",
            )
            submit_btn = gr.Button("Calculate HR & RR Analytics", variant="primary")
        with gr.Column():
            bpm_output = gr.Textbox(label="🌟 Global Estimated Heart Rate (BPM)")
            rr_output = gr.Textbox(label="🌟 Global Estimated Respiratory Rate (BRPM)")

    with gr.Row():
        annotated_output = gr.Video(label="Processed Causal Tracking Video")
        waveform_output = gr.Image(label="Continuous rPPG Waveform Signal")

    submit_btn.click(
        fn=process_video,
        inputs=[video_input],
        outputs=[bpm_output, rr_output, waveform_output, annotated_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

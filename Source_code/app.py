import io
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from inference_utils import predict, visualize
from model import load_model


st.set_page_config(page_title="Crowd Counting", page_icon="👥", layout="wide")


@st.cache_resource(max_entries=2)
def cached_model(path, modified, device_name):
    return load_model(path, torch.device(device_name))


def photo_demo(upload, model, device):
    try:
        rgb = np.array(Image.open(upload).convert("RGB"))
    except Exception as error:
        st.error(f"Gambar tidak bisa dibaca: {error}")
        return
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    with st.spinner("Memproses gambar..."):
        started = time.perf_counter()
        count, density = predict(model, frame, device)
        overlay = visualize(frame, count, density)
    st.metric("Estimasi jumlah orang", f"{count:.1f}")
    st.caption(f"Waktu proses: {time.perf_counter() - started:.2f} detik")
    left, right = st.columns(2)
    left.image(rgb, caption="Gambar asli", use_container_width=True)
    right.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Density heatmap", use_container_width=True)
    success, encoded = cv2.imencode(".jpg", overlay)
    if not success:
        raise OSError("Gagal membuat gambar hasil.")
    st.download_button("Unduh gambar hasil", encoded.tobytes(), "crowd_result.jpg", "image/jpeg", on_click="ignore")
    buffer = io.BytesIO()
    np.save(buffer, density)
    st.download_button("Unduh density map", buffer.getvalue(), "density.npy", "application/octet-stream", on_click="ignore")


def video_demo(upload, model, device):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        source = directory / ("source" + Path(upload.name).suffix.lower())
        source.write_bytes(upload.getvalue())
        capture = cv2.VideoCapture(str(source))
        writer = None
        progress = st.progress(0.0)
        status = st.empty()
        preview = st.empty()
        rows = []
        raw_video = directory / "raw.mp4"
        try:
            if not capture.isOpened():
                raise OSError("Video tidak bisa dibuka.")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("FPS video tidak valid.")
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if writer is None:
                    height, width = frame.shape[:2]
                    output_width, output_height = width + width % 2, height + height % 2
                    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height))
                    if not writer.isOpened():
                        raise OSError("Video writer gagal dibuka.")
                count, density = predict(model, frame, device)
                overlay = visualize(frame, count, density)
                padded = cv2.copyMakeBorder(overlay, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE)
                writer.write(padded)
                index = len(rows)
                rows.append(dict(frame=index, time_seconds=index / fps, estimated_count=count))
                if index % 10 == 0:
                    preview.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=f"Frame {index + 1}", use_container_width=True)
                    status.text(f"Memproses frame {index + 1}" + (f" / {total}" if total > 0 else ""))
                    if total > 0:
                        progress.progress(min((index + 1) / total, 1.0))
        finally:
            capture.release()
            if writer is not None:
                writer.release()
        if not rows:
            raise ValueError("Tidak ada frame yang dapat dibaca.")
        browser_video = directory / "result.mp4"
        status.text("Menyiapkan video hasil...")
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(raw_video), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(browser_video)]
        subprocess.run(command, check=True, capture_output=True)
        video_bytes = browser_video.read_bytes()
        progress.progress(1.0)
        status.success(f"Selesai: {len(rows)} frame")
        preview.empty()
        results = pd.DataFrame(rows)
        left, right = st.columns(2)
        left.metric("Rata-rata orang per frame", f"{results['estimated_count'].mean():.1f}")
        right.metric("Puncak estimasi orang", f"{results['estimated_count'].max():.1f}")
        st.video(video_bytes)
        st.line_chart(results.set_index("time_seconds")[["estimated_count"]])
        st.download_button("Unduh video hasil", video_bytes, "crowd_result.mp4", "video/mp4", on_click="ignore")
        st.download_button("Unduh hitungan per frame", results.to_csv(index=False).encode(), "video_counts.csv", "text/csv", on_click="ignore")


def main():
    st.title("Crowd Counting Demo")
    st.write("Upload gambar atau video untuk melihat estimasi jumlah orang dan density heatmap.")
    with st.sidebar:
        st.header("Model")
        checkpoint_path = st.text_input("Lokasi checkpoint", "outputs/best_csrnet.pth")
        choices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        device_name = st.selectbox("Perangkat", choices, index=len(choices) - 1)
        st.caption("Gunakan checkpoint dari arsitektur notebook atau paket training ini.")
    mode = st.radio("Jenis input", ["Gambar", "Video"], horizontal=True)
    types = ["jpg", "jpeg", "png", "webp"] if mode == "Gambar" else ["mp4", "mov", "avi", "mkv"]
    upload = st.file_uploader("Pilih file", type=types)
    if mode == "Video":
        st.caption("Hitungan dihitung per frame, bukan total pengunjung unik. Video hasil tanpa audio.")
    submitted = st.button("Proses " + mode.lower(), type="primary", disabled=upload is None)
    state_key = (mode, upload.file_id if upload else None, checkpoint_path, device_name)
    if st.session_state.get("input_key") != state_key:
        st.session_state["input_key"] = state_key
        st.session_state["processed_key"] = None
    if submitted:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            st.error("Checkpoint tidak ditemukan. Isi lokasi best_csrnet.pth yang sudah dilatih.")
            return
        try:
            model = cached_model(str(path), path.stat().st_mtime_ns, device_name)
            st.session_state["processed_key"] = state_key
            device = torch.device(device_name)
            if mode == "Gambar":
                photo_demo(upload, model, device)
            else:
                video_demo(upload, model, device)
        except Exception as error:
            st.session_state["processed_key"] = None
            st.error(f"Pemrosesan gagal: {error}")
    elif st.session_state.get("processed_key") == state_key:
        st.info("Klik Proses kembali untuk menampilkan hasil. Unduh hasil sebelum mengubah input.")


if __name__ == "__main__":
    main()

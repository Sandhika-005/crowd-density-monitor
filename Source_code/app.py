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


st.set_page_config(
    page_title="Crowd Density Monitor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# UI styling only: prediction/model functions are kept unchanged.
st.markdown(
    """
    <style>
    :root {
        --cdm-ink: #17243b;
        --cdm-muted: #65738b;
        --cdm-line: #e4eaf3;
        --cdm-primary: #3157d5;
        --cdm-soft: #f4f7fc;
        --cdm-card: #ffffff;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: #f6f8fc;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: var(--cdm-ink);
        letter-spacing: -0.025em;
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: var(--cdm-muted);
    }

    .cdm-hero {
        position: relative;
        overflow: hidden;
        padding: 1.7rem 1.9rem;
        margin-bottom: 1.25rem;
        border: 1px solid #e0e7f3;
        border-radius: 22px;
        background: linear-gradient(120deg, #ffffff 0%, #f1f5ff 70%, #edf2ff 100%);
        box-shadow: 0 10px 30px rgba(27, 48, 91, 0.045);
    }

    .cdm-eyebrow {
        display: inline-block;
        margin-bottom: 0.55rem;
        padding: 0.28rem 0.62rem;
        border: 1px solid #d9e3ff;
        border-radius: 999px;
        background: #edf2ff;
        color: #3157b7;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    .cdm-hero-title {
        margin: 0;
        color: #17243b;
        font-size: clamp(1.8rem, 3vw, 2.55rem);
        font-weight: 780;
        line-height: 1.15;
        letter-spacing: -0.045em;
    }

    .cdm-hero-copy {
        max-width: 760px;
        margin: 0.7rem 0 0 0;
        color: #65738b;
        font-size: 1rem;
        line-height: 1.65;
    }

    .cdm-section-label {
        margin-bottom: 0.2rem;
        color: #3157b7;
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    .cdm-section-copy {
        margin-top: 0;
        margin-bottom: 0.85rem;
        color: #65738b;
        font-size: 0.93rem;
    }

    [data-testid="stSidebar"] {
        background: #f0f3f9;
        border-right: 1px solid #e3e9f2;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #17243b;
    }

    [data-testid="stMetric"] {
        min-height: 112px;
        padding: 1rem 1.05rem;
        border: 1px solid var(--cdm-line);
        border-radius: 16px;
        background: #ffffff;
        box-shadow: 0 5px 18px rgba(27, 48, 91, 0.035);
    }

    [data-testid="stMetricLabel"] {
        color: #65738b;
        font-size: 0.86rem;
    }

    [data-testid="stMetricValue"] {
        color: #17243b;
        font-weight: 750;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e2e8f2;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.88);
        box-shadow: 0 7px 22px rgba(27, 48, 91, 0.035);
    }

    [data-testid="stFileUploaderDropzone"] {
        border: 1.5px dashed #c9d4e8;
        border-radius: 15px;
        background: #fbfcff;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #7d98ef;
        background: #f6f8ff;
    }

    .stButton > button,
    .stDownloadButton > button {
        min-height: 2.65rem;
        border-radius: 11px;
        font-weight: 650;
        transition: all 0.15s ease-in-out;
    }

    .stButton > button[kind="primary"] {
        border: 1px solid #3157d5;
        background: #3157d5;
        color: #ffffff;
    }

    .stButton > button[kind="primary"]:hover {
        border-color: #2546b7;
        background: #2546b7;
        box-shadow: 0 5px 14px rgba(49, 87, 213, 0.18);
    }

    [data-testid="stProgressBar"] > div > div > div > div {
        background: #3157d5;
    }

    hr {
        border-color: #e5eaf3;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .cdm-hero {
            padding: 1.25rem;
            border-radius: 17px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(max_entries=2)
def cached_model(path, modified, device_name):
    return load_model(path, torch.device(device_name))


def photo_demo(upload, model, device):
    """Single-image helper retained for compatibility with the original app."""
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
    right.image(
        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
        caption="Density heatmap",
        use_container_width=True,
    )

    success, encoded = cv2.imencode(".jpg", overlay)
    if not success:
        raise OSError("Gagal membuat gambar hasil.")
    st.download_button(
        "Unduh gambar hasil",
        encoded.tobytes(),
        "crowd_result.jpg",
        "image/jpeg",
        on_click="ignore",
    )

    buffer = io.BytesIO()
    np.save(buffer, density)
    st.download_button(
        "Unduh density map",
        buffer.getvalue(),
        "density.npy",
        "application/octet-stream",
        on_click="ignore",
    )


def multiple_photo_demo(uploads, model, device):
    total_count = 0.0
    processed_count = 0
    failed_count = 0
    summary_rows = []
    summary_area = st.empty()

    st.markdown("## Hasil analisis gambar")
    st.caption(
        "Setiap gambar diproses secara terpisah. Total di bawah merupakan penjumlahan "
        "estimasi tiap gambar, bukan jumlah orang unik."
    )

    for index, upload in enumerate(uploads, start=1):
        with st.container(border=True):
            st.markdown(f"### Gambar {index:02d} · {upload.name}")
            try:
                rgb = np.array(Image.open(upload).convert("RGB"))
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                with st.spinner(
                    f"Memproses gambar {index}/{len(uploads)}: {upload.name}..."
                ):
                    started = time.perf_counter()
                    count, density = predict(model, frame, device)
                    overlay = visualize(frame, count, density)
                    elapsed = time.perf_counter() - started

                total_count += count
                processed_count += 1
                summary_rows.append(
                    {
                        "Nama gambar": upload.name,
                        "Estimasi orang": round(float(count), 1),
                        "Waktu proses (detik)": round(elapsed, 2),
                    }
                )

                metric_col, time_col, resolution_col = st.columns(3)
                metric_col.metric("Estimasi orang", f"{count:.1f}")
                time_col.metric("Waktu proses", f"{elapsed:.2f} dtk")
                height, width = rgb.shape[:2]
                resolution_col.metric("Resolusi", f"{width} × {height}")

                image_col, heatmap_col = st.columns(2, gap="large")
                with image_col:
                    st.image(
                        rgb,
                        caption="Gambar asli",
                        use_container_width=True,
                    )
                with heatmap_col:
                    st.image(
                        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                        caption="Density heatmap",
                        use_container_width=True,
                    )

                success, encoded = cv2.imencode(".jpg", overlay)
                if not success:
                    raise OSError("Gagal membuat gambar hasil.")

                safe_stem = Path(upload.name).stem or f"gambar_{index}"
                download_image_col, download_density_col = st.columns(2)
                with download_image_col:
                    st.download_button(
                        "⬇️ Unduh gambar hasil",
                        encoded.tobytes(),
                        f"{safe_stem}_crowd_result.jpg",
                        "image/jpeg",
                        key=f"image_result_{index}_{upload.file_id}",
                        on_click="ignore",
                        use_container_width=True,
                    )

                buffer = io.BytesIO()
                np.save(buffer, density)
                with download_density_col:
                    st.download_button(
                        "⬇️ Unduh density map (.npy)",
                        buffer.getvalue(),
                        f"{safe_stem}_density.npy",
                        "application/octet-stream",
                        key=f"density_result_{index}_{upload.file_id}",
                        on_click="ignore",
                        use_container_width=True,
                    )

            except Exception as error:
                failed_count += 1
                st.error(f"Gambar ini gagal diproses: {error}")

    if processed_count:
        with summary_area.container():
            st.markdown("## Ringkasan keseluruhan")
            total_col, success_col, failed_col = st.columns(3)
            total_col.metric("Total estimasi orang", f"{total_count:.1f}")
            success_col.metric("Gambar berhasil", f"{processed_count} dari {len(uploads)}")
            failed_col.metric("Gambar gagal", str(failed_count))
            st.caption(
                "Total merupakan penjumlahan estimasi dari gambar yang berhasil diproses. "
                "Jika orang yang sama muncul di beberapa gambar, estimasinya tetap dijumlahkan."
            )
            if summary_rows:
                with st.expander("Lihat ringkasan setiap gambar", expanded=False):
                    st.dataframe(
                        pd.DataFrame(summary_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
    else:
        with summary_area.container():
            st.warning("Tidak ada gambar yang berhasil diproses.")


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
                    writer = cv2.VideoWriter(
                        str(raw_video),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (output_width, output_height),
                    )
                    if not writer.isOpened():
                        raise OSError("Video writer gagal dibuka.")

                count, density = predict(model, frame, device)
                overlay = visualize(frame, count, density)
                padded = cv2.copyMakeBorder(
                    overlay,
                    0,
                    height % 2,
                    0,
                    width % 2,
                    cv2.BORDER_REPLICATE,
                )
                writer.write(padded)
                index = len(rows)
                rows.append(
                    dict(
                        frame=index,
                        time_seconds=index / fps,
                        estimated_count=count,
                    )
                )

                if index % 10 == 0:
                    preview.image(
                        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                        caption=f"Pratinjau frame {index + 1}",
                        use_container_width=True,
                    )
                    status.text(
                        f"Memproses frame {index + 1}"
                        + (f" / {total}" if total > 0 else "")
                    )
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
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            str(raw_video),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(browser_video),
        ]
        subprocess.run(command, check=True, capture_output=True)
        video_bytes = browser_video.read_bytes()
        progress.progress(1.0)
        status.success(f"Selesai diproses · {len(rows)} frame")
        preview.empty()

        results = pd.DataFrame(rows)
        st.markdown("## Ringkasan video")
        with st.container(border=True):
            average_col, peak_col, frames_col = st.columns(3)
            average_col.metric(
                "Rata-rata estimasi per frame",
                f"{results['estimated_count'].mean():.1f}",
            )
            peak_col.metric(
                "Puncak estimasi orang",
                f"{results['estimated_count'].max():.1f}",
            )
            frames_col.metric("Jumlah frame dianalisis", f"{len(rows):,}")
            st.caption(
                "Hitungan video dihitung per frame, bukan jumlah pengunjung unik. "
                "Audio tidak disertakan pada video hasil."
            )

        st.markdown("### Video hasil")
        st.video(video_bytes)

        st.markdown("### Perubahan estimasi sepanjang video")
        st.line_chart(results.set_index("time_seconds")[["estimated_count"]])

        download_video_col, download_csv_col = st.columns(2)
        with download_video_col:
            st.download_button(
                "⬇️ Unduh video hasil",
                video_bytes,
                "crowd_result.mp4",
                "video/mp4",
                on_click="ignore",
                use_container_width=True,
            )
        with download_csv_col:
            st.download_button(
                "⬇️ Unduh hitungan per frame (.csv)",
                results.to_csv(index=False).encode(),
                "video_counts.csv",
                "text/csv",
                on_click="ignore",
                use_container_width=True,
            )


def main():
    st.markdown(
        """
        <div class="cdm-hero">
            <div class="cdm-eyebrow">Computer Vision · Deep Learning</div>
            <h1 class="cdm-hero-title">Crowd Density Monitor</h1>
            <p class="cdm-hero-copy">
                Analisis kerumunan dari gambar atau video. Lihat estimasi jumlah orang,
                visualisasi density heatmap, dan unduh hasil analisis dalam satu tempat.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## ⚙️ Pengaturan")
        st.caption("Atur checkpoint model dan perangkat inferensi.")
        st.divider()
        checkpoint_path = st.text_input(
            "Lokasi checkpoint",
            "outputs/best_csrnet.pth",
            help="Pilih file checkpoint .pth yang kompatibel dengan model proyek ini.",
        )
        choices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        device_name = st.selectbox(
            "Perangkat pemrosesan",
            choices,
            index=len(choices) - 1,
            help="GPU CUDA hanya tersedia jika PyTorch mendeteksinya.",
        )
        if device_name == "cuda":
            st.success("GPU CUDA siap digunakan")
        else:
            st.info("Mode CPU aktif")
        st.divider()
        st.caption(
            "Catatan: estimasi model merupakan perkiraan. Akurasi dapat berubah "
            "sesuai kepadatan, sudut kamera, dan kualitas gambar."
        )

    st.markdown('<div class="cdm-section-label">LANGKAH 01</div>', unsafe_allow_html=True)
    st.markdown("### Pilih jenis analisis")
    st.markdown(
        '<p class="cdm-section-copy">Mulai dengan gambar diam atau video untuk menganalisis kerumunan.</p>',
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Jenis input",
        ["Gambar", "Video"],
        horizontal=True,
        label_visibility="collapsed",
        help="Mode gambar mendukung unggah beberapa gambar sekaligus.",
    )
    types = ["jpg", "jpeg", "png", "webp"] if mode == "Gambar" else ["mp4", "mov", "avi", "mkv"]

    with st.container(border=True):
        st.markdown('<div class="cdm-section-label">LANGKAH 02</div>', unsafe_allow_html=True)
        st.markdown("### Unggah file")
        if mode == "Gambar":
            st.caption("Pilih satu atau beberapa gambar. Format yang didukung: JPG, JPEG, PNG, WEBP.")
        else:
            st.caption("Pilih satu video. Format yang didukung: MP4, MOV, AVI, MKV.")

        upload = st.file_uploader(
            "Pilih file untuk dianalisis",
            type=types,
            accept_multiple_files=(mode == "Gambar"),
            label_visibility="collapsed",
        )

        has_upload = bool(upload) if mode == "Gambar" else upload is not None
        if mode == "Gambar" and has_upload:
            st.success(f"{len(upload)} gambar siap dianalisis")
        elif mode == "Video" and has_upload:
            st.success(f"Video dipilih: {upload.name}")
            st.caption("Hitungan dihitung per frame, bukan jumlah pengunjung unik. Video hasil tanpa audio.")

        button_text = "▶  Proses gambar" if mode == "Gambar" else "▶  Proses video"
        submitted = st.button(
            button_text,
            type="primary",
            disabled=not has_upload,
            use_container_width=True,
        )

    if mode == "Gambar":
        upload_key = tuple((file.file_id, file.name, file.size) for file in upload) if upload else None
    else:
        upload_key = (upload.file_id, upload.name, upload.size) if upload is not None else None

    state_key = (mode, upload_key, checkpoint_path, device_name)
    if st.session_state.get("input_key") != state_key:
        st.session_state["input_key"] = state_key
        st.session_state["processed_key"] = None

    if submitted:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            st.error("Checkpoint tidak ditemukan. Isi lokasi file best_csrnet.pth yang sudah dilatih.")
            return

        try:
            with st.spinner("Memuat model..." ):
                model = cached_model(str(path), path.stat().st_mtime_ns, device_name)
            st.session_state["processed_key"] = state_key
            device = torch.device(device_name)
            st.divider()
            if mode == "Gambar":
                multiple_photo_demo(upload, model, device)
            else:
                st.markdown("## Hasil analisis video")
                video_demo(upload, model, device)
        except Exception as error:
            st.session_state["processed_key"] = None
            st.error(f"Pemrosesan gagal: {error}")
    elif st.session_state.get("processed_key") == state_key:
        st.info("Input sudah pernah diproses. Klik tombol proses kembali untuk menampilkan hasil lagi.")


if __name__ == "__main__":
    main()

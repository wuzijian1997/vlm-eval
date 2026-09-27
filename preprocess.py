"""Step 1: extract frames from SurgCoT videos.

Every video referenced by the given annotation files is uniformly sampled at 1 fps
(configurable) and resized to 210x360 (H x W). Frame k is the frame at timestamp
k / fps seconds and is saved as <out>/<video rel path without .mp4>/<k:06d>.jpg.

Each finished video gets a meta.json; videos that already have one are skipped, so the
script is safe to re-run. At the end, all meta.json files are merged into <out>/index.json.

Example:
    python preprocess.py --annotations Q/test.json Q_dq/test.json --limit 5
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from multiprocessing import Pool

DATA_ROOT = "/data/zijianwu/SurgCoT"
ANNOT_VIDEO_PREFIX = "/data/wanggui/GeneralSurg/"   # absolute prefix used in the HF annotations
VIDEO_ROOT = f"{DATA_ROOT}/GeneralSurg"
FRAME_ROOT = f"{DATA_ROOT}/frames"


def annot_to_local(path):
    """Map an annotation video path to its local path under VIDEO_ROOT."""
    assert path.startswith(ANNOT_VIDEO_PREFIX), path
    return os.path.join(VIDEO_ROOT, path[len(ANNOT_VIDEO_PREFIX):])


def collect_videos(annotation_files):
    videos = set()
    for f in annotation_files:
        path = f if os.path.isabs(f) else os.path.join(DATA_ROOT, f)
        for item in json.load(open(path)):
            videos.update(annot_to_local(v) for v in item["videos"])
    return sorted(videos)


def frame_dir(video, out_root):
    rel = os.path.relpath(video, VIDEO_ROOT)
    return os.path.join(out_root, os.path.splitext(rel)[0])


def probe(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "format=duration:stream=width,height,avg_frame_rate",
         "-of", "json", video],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    stream = info["streams"][0]
    num, den = stream["avg_frame_rate"].split("/")
    return {"duration": float(info["format"]["duration"]),
            "orig_fps": float(num) / float(den) if float(den) else None,
            "orig_width": stream["width"], "orig_height": stream["height"]}


def process(job):
    video, out_root, fps, height, width, quality = job
    dst = frame_dir(video, out_root)
    if os.path.exists(os.path.join(dst, "meta.json")):
        return video, "skipped", None
    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    try:
        meta = probe(video)
        # fps filter: output frame k is the input frame nearest to t = k / fps.
        subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-i", video,
             "-vf", f"fps={fps},scale={width}:{height}:flags=bicubic",
             "-q:v", str(quality), "-start_number", "0",
             os.path.join(tmp, "%06d.jpg")],
            capture_output=True, text=True, check=True)
        num_frames = len([f for f in os.listdir(tmp) if f.endswith(".jpg")])
        if num_frames == 0:
            raise RuntimeError("ffmpeg produced no frames")
        meta.update({"video": video, "frame_dir": dst, "num_frames": num_frames,
                     "sample_fps": fps, "height": height, "width": width})
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        shutil.rmtree(dst, ignore_errors=True)
        os.rename(tmp, dst)
        return video, "done", None
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        msg = e.stderr.strip()[-500:] if isinstance(e, subprocess.CalledProcessError) else str(e)
        return video, "failed", msg


def write_index(videos, out_root):
    index, missing = {}, []
    for v in videos:
        meta_path = os.path.join(frame_dir(v, out_root), "meta.json")
        if os.path.exists(meta_path):
            index[os.path.relpath(v, VIDEO_ROOT)] = json.load(open(meta_path))
        else:
            missing.append(v)
    with open(os.path.join(out_root, "index.json"), "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    return index, missing


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", nargs="+", default=["Q/test.json", "Q_dq/test.json"],
                   help="annotation files (relative to %s) whose videos are processed" % DATA_ROOT)
    p.add_argument("--out", default=FRAME_ROOT)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--height", type=int, default=210)
    p.add_argument("--width", type=int, default=360)
    p.add_argument("--quality", type=int, default=2, help="ffmpeg JPEG -q:v (2 = high, 31 = low)")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="only process the first N videos (debugging)")
    args = p.parse_args()

    videos = collect_videos(args.annotations)
    if args.limit:
        videos = videos[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(videos)} videos -> {args.out} ({args.fps} fps, {args.height}x{args.width})", flush=True)

    jobs = [(v, args.out, args.fps, args.height, args.width, args.quality) for v in videos]
    counts, failures, t0 = {"done": 0, "skipped": 0, "failed": 0}, [], time.time()
    with Pool(args.workers) as pool:
        for i, (video, status, err) in enumerate(pool.imap_unordered(process, jobs), 1):
            counts[status] += 1
            if status == "failed":
                failures.append({"video": video, "error": err})
                print(f"FAILED {video}: {err}", flush=True)
            if i % 50 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {counts}  {time.time() - t0:.0f}s", flush=True)

    index, missing = write_index(videos, args.out)
    if failures:
        with open(os.path.join(args.out, "failures.json"), "w") as f:
            json.dump(failures, f, ensure_ascii=False, indent=1)
    total = sum(m["num_frames"] for m in index.values())
    print(f"index.json: {len(index)} videos, {total} frames; {len(missing)} missing/failed", flush=True)


if __name__ == "__main__":
    main()

<h1 align="center">Cricket Biomechanics Analyzer</h1>

<p align="center">
  <b>Real-time batting technique analysis powered by computer vision</b><br>
  <sub>Detect. Measure. Coach. -- all from a single camera angle.</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLO11--Pose-blue?style=for-the-badge" alt="YOLO11-Pose"/>
  <img src="https://img.shields.io/badge/Backend-Apple_MPS-black?style=for-the-badge&logo=apple" alt="Apple MPS"/>
  <img src="https://img.shields.io/badge/FPS-20--30+-green?style=for-the-badge" alt="FPS"/>
  <img src="https://img.shields.io/badge/Metrics-14+-orange?style=for-the-badge" alt="Metrics"/>
  <img src="https://img.shields.io/badge/Python-3.13-yellow?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
</p>

---

## What is this?

A real-time system that takes **live webcam** or **recorded video** of a batsman, runs pose estimation using **YOLO11-Pose**, and overlays:

- **14+ biomechanical metrics** (weight distribution, head position, knee angles, spine tilt, bat angle...)
- **Shot type classification** (Drive, Pull, Cut, Sweep, Defense)
- **Plain-English coaching insights** directly on the video frame

No force plates. No lab equipment. Just a camera and a laptop.

---

## Demo

```
+------------------------------------------------------------------+
| RIGHT-HAND         FRONT FOOT DRIVE  82%              26 FPS     |
|                                                  +---------------+
|         [ neon skeleton ]                        | BIOMECHANICS  |
|              ( CENTER )                          | * Head    Ctr |
|                                                  | * Tilt   2.1  |
|            145         138                       | * L Knee  142 |
|                                                  | * R Knee  138 |
|       [bat shaft + swing trail]                  | * Spine  8.3  |
|                                                  | * Shldr  3.1  |
|                                                  | * Hips   2.4  |
|                                                  |--- BAT -------|
|                                                  | * Face Strt   |
|                                                  | * Angle  12   |
|                                                  +---------------+
|                 WEIGHT DISTRIBUTION                              |
|  BACK  35% [======|======================] 65%  FRONT            |
|                                                                  |
|  [OK] Weight balanced (65% front)                                |
|  [! ] Slightly stiff legs -- try a deeper knee bend              |
|  [TIP] Drive: keep the elbow high and follow through             |
+------------------------------------------------------------------+
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/PhaeezAli/cricket_biomech.git
cd cricket_biomech

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run on a video file
python app/main.py --source file --input path/to/batting_video.mp4

# 4. Run on webcam
python app/main.py --source webcam
```

> YOLO11-Pose model (`yolo11n-pose.pt`, ~6MB) auto-downloads on first run.

---

## Usage

```bash
# Basic video analysis
python app/main.py --source file --input data/input_videos/batting.mp4

# Save annotated output
python app/main.py --source file --input batting.mp4 --output annotated.mp4

# Live webcam
python app/main.py --source webcam

# Specify handedness (default: auto-detected)
python app/main.py --source webcam --handedness left

# Use a custom YOLO bat detection model
python app/main.py --source file --input batting.mp4 --bat-model path/to/best.pt
```

**Keyboard controls:**
| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save snapshot of current frame |

---

## Architecture

```
Video Frame
    |
    v
+---------------------------+
|   YOLO11-Pose (on MPS)    |  Single model: detection + 17 keypoints
+---------------------------+
    |                |
    v                v
+----------+   +----------------+
|   Bat    |   | Biomechanics   |
| Detector |   | Engine         |
| (geom /  |   | - weight bal.  |
|  YOLO)   |   | - head pos.    |
+----------+   | - joint angles |
    |          +----------------+
    |                |
    v                v
+---------------------------+
|    Shot Classifier        |  30-frame rolling buffer + rule engine
+---------------------------+
    |
    v
+---------------------------+
|    Insights Engine        |  Plain-English coaching tips
+---------------------------+
    |
    v
+---------------------------+
|    Visualizer             |  Neon skeleton, gauges, panels
+---------------------------+
```

---

## What It Measures

### Body Metrics
| Metric | Method |
|--------|--------|
| Front / back foot weight % | Hip centre-of-mass position relative to ankles |
| Head position (Front / Center / Back) | Ear midpoint plumb line vs feet center |
| Head tilt | Ear-to-ear line angle from horizontal |
| Knee flexion (both legs) | 3-point joint angle: hip -- knee -- ankle |
| Spine tilt | Hip midpoint to shoulder midpoint vs vertical |
| Shoulder alignment | Shoulder line angle from horizontal |
| Hip alignment | Hip line angle from horizontal |

### Bat Metrics
| Metric | Method |
|--------|--------|
| Bat face (Straight / Angled / Cross-bat) | Elbow-wrist-blade vector vs vertical |
| Bat swing phase (Backswing / Downswing / Set) | Y-velocity of bat head over frames |
| Backswing height (0-150%) | Bat head height relative to body height |

### Shot Classification
| Shot | Key Triggers |
|------|-------------|
| Front Foot Drive | Front weight > 65%, knees extended |
| Pull Shot | Back weight > 60%, wrists above shoulders |
| Cut Shot | Back weight > 55%, lateral wrist displacement |
| Sweep | Front weight > 60%, deep front knee bend |
| Back Foot Defense | Back weight > 55%, upright spine |

---

## Project Structure

```
cricket_biomech/
|-- app/
|   |-- main.py                 # CLI entry point
|   |-- pose_estimator.py       # YOLO11-Pose wrapper
|   |-- bat_detector.py         # Geometric + optional YOLO bat detection
|   |-- shot_classifier.py      # Rule-based shot type classifier
|   |-- insights.py             # Coaching tips engine
|   |-- visualizer.py           # All overlay rendering
|   |-- video_input.py          # Webcam / file video source
|   +-- biomechanics/
|       |-- __init__.py         # analyse() orchestrator
|       |-- metrics.py          # BiomechanicsResult dataclass
|       |-- weight_balance.py   # CoM-based weight distribution
|       |-- head_position.py    # Plumb-line head analysis
|       +-- angles.py           # Joint angle calculations
|-- data/
|   |-- input_videos/           # Place your batting videos here
|   +-- output_videos/          # Annotated output saved here
|-- requirements.txt
+-- README.md
```

---

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Pose Estimation | **YOLO11-Pose** | Single model for detection + pose in one pass |
| GPU Backend | **PyTorch MPS** | Native Apple Silicon acceleration |
| Video I/O | **OpenCV** | Industry standard, webcam + file support |
| Math | **NumPy** | Vector math, angle calculations |

---

## How Bat Detection Works

**Without any custom model** -- the system uses geometric inference:

1. Bat grip = midpoint of both wrists
2. Bat direction = extends the elbow-to-wrist vector past the hands
3. Bat length = ~50% of the person's pixel height (auto-scales)
4. Swing phase = tracked from y-velocity of the bat head over 45 frames

For higher accuracy, train a custom YOLO model on a cricket bat dataset and pass it with `--bat-model`.

---

## Roadmap

- [ ] Lead elbow angle tracking (top-arm shoulder-elbow-wrist)
- [ ] Head drift velocity (movement trend over time, not just position)
- [ ] Hip-shoulder separation angle (power indicator for drives)
- [ ] Session summary report (per-shot PDF/CSV export)
- [ ] Custom YOLO bat model trained on cricket dataset
- [ ] Streamlit dashboard mode

---

## Requirements

- Python 3.10+
- macOS with Apple Silicon (M1/M2/M3/M4) for MPS acceleration
- Also works on CPU (slower) or NVIDIA GPU (CUDA)
- Webcam or video file of batting from a **side-on angle**

---



<p align="center">
  <sub>Built with YOLO11-Pose, OpenCV, PyTorch, and NumPy</sub>
</p>

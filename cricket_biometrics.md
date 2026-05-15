<h1 align="center">
  <br>
  Cricket Biomechanics Analyzer
  <br>
</h1>

<p align="center">
  <b>Real-time batting technique analysis using computer vision</b><br>
  <sub>YOLO11-Pose | Apple MPS | OpenCV | Python</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLO11--Pose-blue?style=for-the-badge" alt="YOLO11-Pose"/>
  <img src="https://img.shields.io/badge/Backend-Apple_MPS-black?style=for-the-badge&logo=apple" alt="Apple MPS"/>
  <img src="https://img.shields.io/badge/FPS-20--30+-green?style=for-the-badge" alt="FPS"/>
  <img src="https://img.shields.io/badge/Metrics-14+-orange?style=for-the-badge" alt="Metrics"/>
  <img src="https://img.shields.io/badge/Python-3.13-yellow?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
</p>

---

## The Pitch

> *"I built a real-time cricket batting analysis system that takes live video or a recorded batting session, detects the batsman's body using computer vision, and automatically measures key biomechanical metrics -- like weight distribution, head position, knee angles, and bat swing path -- then overlays plain-English coaching tips directly onto the video in real time."*

---

## Tech Stack

<table>
  <tr>
    <th>Layer</th>
    <th>Tool</th>
    <th>Why</th>
  </tr>
  <tr>
    <td><b>Pose Estimation</b></td>
    <td><code>YOLO11-Pose</code> (Ultralytics)</td>
    <td>Single model for detection + 17-keypoint pose in one forward pass</td>
  </tr>
  <tr>
    <td><b>GPU Backend</b></td>
    <td><code>PyTorch MPS</code></td>
    <td>Native Apple Silicon GPU acceleration -- no CUDA needed</td>
  </tr>
  <tr>
    <td><b>Video I/O</b></td>
    <td><code>OpenCV</code></td>
    <td>Industry standard, handles both file and webcam sources</td>
  </tr>
  <tr>
    <td><b>Math</b></td>
    <td><code>NumPy</code></td>
    <td>Angle calculations, vector math, coordinate transforms</td>
  </tr>
  <tr>
    <td><b>Language</b></td>
    <td><code>Python 3.13</code></td>
    <td>Rapid prototyping, rich CV/ML ecosystem</td>
  </tr>
</table>

---

## Architecture

```
                          +------------------+
                          |   Video Input    |
                          |  webcam / file   |
                          +--------+---------+
                                   |
                                   v
                    +------------------------------+
                    |     YOLO11-Pose (on MPS)     |
                    |  detection + 17 keypoints    |
                    +------------------------------+
                          |                |
                          v                v
              +-----------+----+    +------+----------+
              |  Bat Detector  |    | Biomechanics    |
              |  (geometric /  |    | Engine          |
              |   YOLO model)  |    |                 |
              +-------+--------+    | - weight bal.   |
                      |             | - head pos.     |
                      |             | - joint angles  |
                      |             +------+----------+
                      |                    |
                      +--------+-----------+
                               |
                               v
                    +----------+----------+
                    |  Shot Classifier     |
                    |  30-frame rolling    |
                    |  buffer + rules     |
                    +----------+----------+
                               |
                               v
                    +----------+----------+
                    |  Insights Engine     |
                    |  plain-English tips  |
                    |  severity-ranked     |
                    +----------+----------+
                               |
                               v
                    +----------+----------+
                    |  Visualizer          |
                    |  annotated output    |
                    +---------------------+
```

---

## What It Measures

<table>
  <tr>
    <th align="left">Category</th>
    <th align="left">Metric</th>
    <th align="left">How</th>
  </tr>
  <tr><td colspan="3"><b>Body</b></td></tr>
  <tr>
    <td>Balance</td>
    <td>Front / back foot weight %</td>
    <td>Hip CoM position relative to ankles</td>
  </tr>
  <tr>
    <td rowspan="2">Head</td>
    <td>Position (Front / Center / Back)</td>
    <td>Ear midpoint plumb line vs feet center</td>
  </tr>
  <tr>
    <td>Tilt</td>
    <td>Ear-to-ear line angle from horizontal</td>
  </tr>
  <tr>
    <td>Knees</td>
    <td>Left & right flexion angle</td>
    <td>3-point joint angle: hip-knee-ankle</td>
  </tr>
  <tr>
    <td rowspan="3">Posture</td>
    <td>Spine tilt</td>
    <td>Hip midpoint to shoulder midpoint vs vertical</td>
  </tr>
  <tr>
    <td>Shoulder alignment</td>
    <td>Shoulder line angle from horizontal</td>
  </tr>
  <tr>
    <td>Hip alignment</td>
    <td>Hip line angle from horizontal</td>
  </tr>
  <tr><td colspan="3"><b>Bat</b></td></tr>
  <tr>
    <td>Bat face</td>
    <td>Straight / Angled / Cross-bat / Horizontal</td>
    <td>Elbow-wrist-blade vector vs vertical</td>
  </tr>
  <tr>
    <td>Swing phase</td>
    <td>Backswing / Downswing / Set</td>
    <td>Y-velocity of bat head over frame history</td>
  </tr>
  <tr>
    <td>Backswing height</td>
    <td>0-150%</td>
    <td>Bat head height relative to body height</td>
  </tr>
  <tr><td colspan="3"><b>Shot</b></td></tr>
  <tr>
    <td>Shot type</td>
    <td>Drive / Pull / Cut / Sweep / Defense</td>
    <td>Rule-based classifier over 30-frame buffer</td>
  </tr>
</table>

---

## Key Technical Decisions

<details>
<summary><b>Why YOLO11-Pose instead of YOLO + MediaPipe?</b></summary>
<br>
The original prototype used YOLO for detection and MediaPipe for pose separately -- two model loads, two inference calls per frame, and coordinate remapping between crops. YOLO11-Pose does both in one pass, reducing latency and code complexity. On Apple MPS it runs at 20-30+ FPS.
</details>

<details>
<summary><b>How do you calculate weight distribution without a force plate?</b></summary>
<br>
We use the hip midpoint as a Centre of Mass proxy. Since the hips carry most of the body's mass, their horizontal position relative to the two ankles gives a reliable directional estimate of weight shift. Normalised as:
<br><br>
<code>front_pct = (CoM_x - back_ankle_x) / (front_ankle_x - back_ankle_x) x 100</code>
</details>

<details>
<summary><b>How does bat detection work without a trained model?</b></summary>
<br>
Geometric inference: the handle sits between the two wrists. The blade extends in the same direction as the elbow-to-wrist vector, past the grip. Bat length is estimated as ~50% of the person's pixel height. A custom YOLO model can be plugged in with <code>--bat-model</code> for higher accuracy.
</details>

<details>
<summary><b>How is handedness auto-detected?</b></summary>
<br>
Over the first 10 frames, we compare left ankle x vs right ankle x. A right-handed batsman in a side-on view has their left foot (front foot) at a lower x value. Majority vote across 10 frames, then locked for the session.
</details>

<details>
<summary><b>How do you prevent coaching insights from flickering?</b></summary>
<br>
The ShotClassifier uses a 30-frame rolling buffer with majority voting -- a shot label only fires if >50% of recent frames match the rule. Insights inherit this stability.
</details>

---

## Challenges & Solutions

| Challenge | Solution |
|---|---|
| MPS tensors can't be passed to NumPy | Added `.cpu()` conversion before `np.argmax` |
| Overlay text flickering every frame | Rolling-buffer majority voting for shot type + severity-sorted insights |
| No force plate available | Hip CoM proxy -- directionally accurate, matches visual coaching assessment |
| Camera angle dependency | Designed for side-on view (standard cricket coaching angle) |

---

## Roadmap

- [ ] Lead elbow angle (shoulder-elbow-wrist for top arm)
- [ ] Head drift velocity (movement trend, not just static position)
- [ ] Hip-shoulder separation angle (power indicator for drives)
- [ ] Session report -- per-shot summary PDF after a batting session
- [ ] Custom YOLO bat model trained on cricket-specific dataset

---

## Resume Bullet Points

> Pick 2-3 depending on the role you are applying for.

### For ML / Computer Vision roles

- Built a **real-time cricket biomechanics analyzer** using **YOLO11-Pose** on **Apple MPS**, extracting 14+ biomechanical metrics from video at **20-30 FPS**
- Engineered a **geometric bat-tracking algorithm** that infers bat position, angle, and swing phase from body keypoints -- without requiring a separately trained detection model
- Designed a **modular computer vision pipeline** (OpenCV, PyTorch, Ultralytics) covering pose estimation, biomechanics computation, shot classification, and real-time annotated video overlay

### For Data / Analytics roles

- Developed a **rule-based sports analytics engine** that classifies cricket shot types (drive, pull, cut, sweep, defense) using a **30-frame rolling biomechanics buffer** with majority-vote smoothing
- Converted raw pose keypoints into **14+ actionable coaching metrics** (weight distribution, joint angles, spine tilt, bat face angle) and surfaced them as **plain-English real-time feedback**

### For Full-Stack / Product roles

- Built an **end-to-end AI sports analysis tool** from scratch -- CV model integration, custom analytics engine, and real-time annotated video output -- deployable via CLI on any Mac laptop
- Designed a **coaching insight system** that translates raw biomechanical data into context-aware, shot-specific feedback understandable by athletes with no technical background

---

<p align="center">
  <sub>Built with Python, YOLO11-Pose, OpenCV, and NumPy</sub>
</p>

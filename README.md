# 🎮 DQN for Atari Breakout

## 👤 Students Details

* **Name:** Sanskriti Singh
* **Roll No:** 23BAI10269

* **Name:** Divya Shukla
* **Roll No:** 23BAI11320

---

## 📌 Project Description

This project implements a **Deep Q-Network (DQN)** to play the Atari game **Breakout** using Reinforcement Learning.

The agent learns to play the game by interacting with the environment and improving its performance over time using Q-learning.

---

## 🧠 Key Concepts Used

* Reinforcement Learning
* Deep Q-Network (DQN)
* Experience Replay Buffer
* Epsilon-Greedy Strategy
* Target Network

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Gymnasium (Atari Environment)
* OpenCV
* NumPy

---

## ⚙️ How It Works

1. The environment is preprocessed (grayscale + resize + frame stacking)
2. The DQN model predicts Q-values for actions
3. Agent selects actions using epsilon-greedy policy
4. Experiences are stored in replay buffer
5. Model is trained using sampled experiences
6. Target network stabilizes learning

---

## ▶️ Output

* Training reward vs episodes graph
* Loss curve
* Gameplay simulation of trained agent
* Generated gameplay video (`breakout_dqn.mp4`)

---

## 📂 File Included

* `Lily_23BCS123.py` → Main implementation file

---

## 📌 Note

* Designed to run in environments supporting Gymnasium Atari
* GPU support is used if available

---

<p align="center">
  <a href="https://www.uit.edu.vn/" title="University of Information Technology" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="University of Information Technology (UIT)">
  </a>
</p>

<h1 align="center"><b>CS221.Q12 - Natural Language Processing</b></h1>

# CS221 Course Project: Vietnamese Named Entity Recognition (NER)

> This repository contains the full implementation of a **Vietnamese Named Entity Recognition (NER)** system developed for the course **CS221.Q12 – Natural Language Processing** at the University of Information Technology (UIT – VNU-HCM).  
>  
> The project focuses on **sequence labeling** for Vietnamese text using three classical and neural approaches: **Hidden Markov Model (HMM)**, **Conditional Random Fields (CRF)**, and **BiLSTM-CRF**.  
>  
> In addition to model training and evaluation, we also provide a **Flask-based interactive demo** that allows users to switch between CRF and BiLSTM-CRF models and visualize prediction results.

---

## Team Information
| No. | Student ID | Full Name | Role | Github | Email |
|----:|:----------:|-----------|------|--------|-------|
| 1 | 23521143 | Nguyen Cong Phat | Leader | [paht2005](https://github.com/paht2005) | 23521143@gm.uit.edu.vn |
| 2 | 23521168 | Nguyen Le Phong | Member | [kllp031](https://github.com/kllp031) | 23521168@gm.uit.edu.vn  |
| 3 | 23520384 | Pham Tran Khanh Duy | Member |  |  23520384@gm.uit.edu.vn  | 


---

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Demo](#demo)
- [Conclusion](#conclusion)
- [License](#license)

---

## Features
- Implementation of **three NER models**:
  - Hidden Markov Model (HMM)
  - Conditional Random Fields (CRF)
  - BiLSTM-CRF (PyTorch)
- Full training pipeline using **Jupyter notebooks**.
- Evaluation using **token-level and span-level metrics**.
- Saved best models for inference (`.joblib`, `.pt`).
- **Flask web demo**:
  - Switch between CRF and BiLSTM-CRF.
  - Highlight predicted named entities.
  - Display model metrics (F1, Precision, Recall).
- Clean project structure suitable for academic submission.

---

## Dataset
- Dataset: **VLSP 2016 Vietnamese NER**
- Format: CoNLL-style text files.
- Data splits:
  - `train.txt`
  - `test.txt`
- Entity types include:
  - `PER` (Person)
  - `ORG` (Organization)
  - `LOC` (Location)
  - `MISC`

---

## Repository Structure


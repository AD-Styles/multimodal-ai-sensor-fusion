# 🌐 Multimodal AI: Sensor Fusion to Contrastive Pre-training

### RGB 카메라와 LiDAR 센서 데이터를 결합하여, 조기·후기·중간 융합 아키텍처를 비교하고 CLIP 스타일 대조학습과 크로스모달 프로젝션까지 단계별로 구현한 멀티모달 AI 포트폴리오.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![NVIDIA DLI](https://img.shields.io/badge/NVIDIA-DLI_Based-76b900.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-f7931e.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 프로젝트 요약 (Project Overview)

딥러닝을 공부하다 보면 처음에는 이미지 하나, 텍스트 하나처럼 단일 종류의 데이터만 다루는 모델을 접하게 됩니다. 그런데 현실의 자율주행 자동차나 의료 진단 시스템은 카메라만이 아니라 LiDAR 센서, 음성, CT 스캔 같은 여러 종류의 정보를 동시에 받아 판단을 내립니다. 이 프로젝트는 "서로 다른 종류의 데이터를 신경망이 어떻게 함께 이해할 수 있는가?"라는 질문에서 출발했습니다.

NVIDIA DLI 멀티모달 AI 강좌의 핵심 개념을 직접 구현하면서, 단순히 코드를 옮겨 적는 것이 아니라 합성 데이터를 직접 설계하고 다섯 가지 모델 구조를 처음부터 만들어 비교했습니다. 각 융합 방식이 서로 다른 상황에서 어떤 강점을 보이는지 수치로 확인하고, 나아가 OpenAI CLIP의 핵심 아이디어인 대조학습을 FashionMNIST와 소벨 에지 검출을 결합해 직접 재현했습니다. 마지막으로는 LiDAR 임베딩 공간을 RGB 임베딩 공간으로 옮겨주는 크로스모달 프로젝터까지 구현하며, 멀티모달 AI 시스템의 전체 흐름을 하나의 파이프라인으로 연결했습니다.

---

## 📂 프로젝트 구조 (Project Structure)

```text
multimodal-ai-sensor-fusion/
├── plots/                          # 메인 코드 실행 시 생성되는 시각화 결과물
├── main.py                         # 전체 파이프라인 통합 실행 스크립트
├─ .gitignore                      
├─ LICENSE                         
├─ README.md                # 프로젝트 개요 및 가이드 문서
└─ requirements.txt         # 핵심 라이브러리 목록
```


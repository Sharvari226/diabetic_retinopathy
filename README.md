This repository presents a clinically-focused deep learning framework for Diabetic Retinopathy (DR) screening, emphasizing reduction of false positives through high specificity and robust image quality handling.

**Highlights:**

**Two-Stage Pipeline:**

1)**Image Quality Assessment (ResNet18 on DRIMDB dataset)** → Filters blurry/poor-quality fundus images (97% validation accuracy) to prevent artifact-induced false positives.

2)**DR Severity Grading (ResNet50 + CBAM attention on APTOS subset, 1751 images)** → 5-class classification (0: No DR → 4: Proliferative DR) with ~87% accuracy and ~99% specificity on healthy retinas.

**Key Benefits**: Minimizes unnecessary referrals, improves operational efficiency, and includes Grad-CAM for lesion-focused explainability.

**Tech:** PyTorch, Albumentations, OpenCV – Fully reproducible in Google Colab.

**Flowchart:**
<img width="2207" height="3996" alt="Untitled diagram-2026-01-05-074514" src="https://github.com/user-attachments/assets/fa50e956-7044-48f9-a2b1-b03e1c869542" />

# ADAS-CV: road hazard and obstacle classifier (CIFAR-10)

ADAS-CV is a student prototype of the image-recognition step inside a driver-assistance system. A ResNet-50 network, pre-trained on ImageNet and fine-tuned on CIFAR-10, classifies an uploaded picture into one of 10 classes. The app then maps the class to an alert level: RED for living things, YELLOW for vehicles, BLUE for low-risk objects, or UNKNOWN when the model is not sure.

This is a learning project. It is not a safety device and must not be used in a vehicle.

**Live app:** https://techcrushgroup27aimlclassa.streamlit.app

**Full report:** [`docs/adas_cv_capstone_report.docx`](docs/adas_cv_capstone_report.docx)

Group 27, AI/ML Class A, TechCrush. October 2026.

## Results

| Measure | Value |
|---|---|
| Test accuracy (10 classes, 10,000 test images) | 94.70% |
| Macro precision, recall, F1-score | 0.947, 0.947, 0.947 |
| Accuracy at the alert level (3 groups) | 99.05% |
| Time per prediction (Colab CPU / Tesla T4 GPU) | 121 ms / 7.8 ms |
| Accuracy with added noise (QA test C4) | 9.4% |
| Accuracy with blur (QA test C1) | 26.3% |

The QA tests show clear limits. The model has no class for pedestrians, cyclists or motorcycles, it is confidently wrong on noisy and blurred pictures, and a street scene was labelled a plane with 96.2% confidence. Chapter 4 of the report gives every table and chart, and `docs/project_overview.md` has a short summary.

## How to run the app on your own computer

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The app loads `app/resnet50_cifar10.pth`. The file `app/cifar10_resnet18_adas.pth` is an earlier ResNet-18 baseline and is not used.

## Repository structure

```
cifar10-animal-object-recognizer/
├── app/          Streamlit app and the trained ResNet-50 weights
├── docs/         Architecture notes and project overview
├── notebooks/    Colab notebooks (training and QA)
├── results/      Metrics, charts and QA output files
├── src/          Python modules and scripts
└── requirements.txt
```

## QA

The notebook `notebooks/02_qa_evaluation.ipynb` reloads the saved model, evaluates it on the 10,000 CIFAR-10 test images, and runs seven image-quality tests. Its output files are in `results/`.

## Team

Group 27, AI/ML Class A, TechCrush. Team lead: Sanni Oluwadarasimi Peter.
Tutors and supervisors: Miracle Nwadiora and Lamzey Aribido.

## Team workflow

Branches, commits and file naming rules are in `CONTRIBUTING.md`.

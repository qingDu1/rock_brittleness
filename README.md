# Rock Brittleness Analysis System

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [Output Description](#output-description)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## Project Overview
This system analyzes rock brittleness characteristics using a multimodal data fusion approach, combining SEM images, EDS elemental analysis, and XRD mineral composition data for comprehensive analysis.

### Key Features
- Multimodal data processing and feature extraction
- Feature alignment based on optimal transport
- Deep learning model training and prediction
- Feature importance analysis
- Result visualization

## Installation

1. Clone the repository:
```bash
git clone git clone https://github.com/your-repo/rock-brittleness.git

cd rock-brittleness
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
Create the following data structure in the project root directory:
```
data/
├── image/              # Rock SEM images
├── EDS_DATA.xlsx      # EDS elemental analysis data
├── XRD_data.xlsx      # XRD mineral composition data
└── Brittleness_DATA.xlsx    # Brittleness index data
```

### 2. Model Training
Execute the training script:
```bash
python run_train.py
```

Training process will automatically create an experiment record directory: `experiments/20250214_151055/`

### 3. Model Prediction
Use the trained model for prediction:
```bash
python run_predict.py --model experiments/timestamp/best_model.pth --data /path/to/new/data
```

## Data Requirements

### Data Specifications
| Type | Format Requirements | Description |
|------|----------|------|
| SEM Images | PNG, JPG, JPEG, TIF | Consistent resolution recommended, filenames should match sample IDs |
| EDS Data | Excel(.xlsx) | Required columns: Sample_ID, O, Na, Mg, Al, Si, K, Ca, Fe |
| XRD Data | Excel(.xlsx) | Required columns: Sample_ID + mineral composition columns |
| Brittleness Index | Excel(.xlsx) | Required columns: Sample_ID, Brittleness |

## Output Description

The training process generates the following in `experiments/timestamp/`:

- `best_model.pth`: Best model weights
- `training_history.png`: Training process visualization
- `predictions.png`: Prediction results visualization
- `feature_importance.csv`: Feature importance analysis
- `correlation_heatmap.png`: Feature correlation heatmap
- `analysis_report.json`: Analysis report

## Model Performance

Typical model performance metrics:
- R² Score: > 0.90
- RMSE: < 0.05
- MAE: < 0.04

## Troubleshooting

### Common Errors and Solutions

1. Data Loading Error
```
Error: FileNotFoundError: [Errno 2] No such file or directory
Solution: Check if data file paths are correct
```

2. GPU Memory Error
```
Error: RuntimeError: CUDA out of memory
Solution: Reduce batch size or use CPU for training
```

3. Data Format Error
```
Error: KeyError: 'Sample_ID'
Solution: Verify column names in Excel files
```

## FAQ

### Q: How to choose an appropriate learning rate?
A: Recommended to start with 0.001 and adjust based on training curves.

### Q: How long does training typically take?
A: Varies with data volume and hardware configuration, typically ranging from a few to dozens of hours.

### Q: How to improve model performance?
A: Try the following methods:
- Increase training data
- Adjust model parameters
- Optimize feature extraction
- Use data augmentation

## Important Notes

- Back up data before processing
- Monitor GPU memory usage
- Regularly validate result accuracy
- Keep dependencies updated

## Contact

For questions, please reach out through:
- Submit an Issue
- Email: your.email@example.com

## License

This project is licensed under the MIT License


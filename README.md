# BVA-SAT: Behavioral Vulnerability Assessment Framework for IoT-Based Satellite Security

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)](https://www.tensorflow.org/)

##  Overview

BVA-SAT is a comprehensive five-phase framework that integrates CVE-based vulnerability assessment with behavioral anomaly detection for IoT-enabled satellite systems. This framework addresses the unique security challenges of satellite environments by combining traditional cybersecurity approaches with satellite-specific telemetry analysis.

##  Key Features

- **Five-Phase Assessment Framework**: Conceptual pipeline covering asset discovery, vulnerability scanning, business assessment, remediation, and behavioral analysis
- **ML-Based Detection**: Neural networks (standard, deep, attention-based) and classical ML models (Random Forest, Gradient Boosting)
- **Satellite-Specific Design**: Tailored for IoT satellite telemetry including SNR, throughput, magnetic field, and power metrics
- **CVE Integration**: Maps vulnerabilities to 6 satellite attack categories (DoS, Jamming, Spoofing, Replay, Eavesdropping, MITM)
- **Behavioral Baselines**: Statistical anomaly detection using 2σ thresholds

##  Installation
```bash
# Clone the repository
git clone https://github.com/Osama-Abdelhameed/BVA-SAT-FRAMEWORK.git
cd BVA-SAT-FRAMEWORK

# Install required packages
pip install -r requirements.txt
```

## Files Description

- **`bva_sat_implementation.py`**: Main framework implementation with 5-phase assessment pipeline
- **`bva_sat_benchmark.py`**: Benchmark evaluation script for UNSW-NB15, CIC-IDS2017, and UNSW-IoTSAT datasets
- **`inventory.yaml`**: Asset inventory configuration for the UNSW-IoTSAT testbed
- **`requirements.txt`**: Python dependencies


##  Dataset Setup & Usage

### Required Directory Structure
```
project_folder/
├── bva_sat_implementation.py
├── bva_sat_benchmark.py
├── inventory.yaml
├── requirements.txt
├── README.md
└── data/                    # Create this folder
    ├── UNSW-NB15/          # Download from UNSW
    ├── CICDataset/         # Download CIC-IDS2017
    └── UNSW-IoTSAT/        # Download IoTSAT dataset
```
### Step 1: Download Datasets

The framework supports three datasets:

1. **UNSW-NB15**: 2.54M samples, 49 features, network intrusion benchmark
   - Download: [UNSW Website](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
   - Download all 4 CSV files (UNSW-NB15_1.csv to UNSW-NB15_4.csv)
   - Download NUSW-NB15_features.csv (column names)
   - Place in `data/UNSW-NB15/`

2. **CIC-IDS2017**: 2.83M samples, 78 features, enterprise/IoT attacks
   - Download: [UNB Website](https://www.unb.ca/cic/datasets/ids-2017.html)
   - Download MachineLearningCVE.zip
   - Extract to `data/CICDataset/MachineLearningCSV/MachineLearningCVE/`

3. **UNSW-IoTSAT**: 0.40M samples, 35 features, satellite telemetry
   - Available: [GitHub](https://github.com/Osama-Abdelhameed/UNSW-IoTSAT)
   - Download UNSW_IoTSAT_With_Feature_Engineering.csv
   - Place in `data/UNSW-IoTSAT/`

### Step 2: Run the Framework

#### Option A: Default data directory (./data)
```bash
python bva_sat_implementation.py
python bva_sat_benchmark.py
```

#### Option B: Custom data directory
```bash
python bva_sat_implementation.py --data-dir /your/path/to/datasets
python bva_sat_benchmark.py --data-dir /your/path/to/datasets
```

#### Option C: Environment variable
```bash
export BVA_SAT_DATA_DIR=/your/path/to/datasets
python bva_sat_implementation.py
python bva_sat_benchmark.py
```

### Expected Output

**bva_sat_implementation.py** will:
1. Load assets from inventory.yaml
2. Scan for vulnerabilities (uses real CVE data)
3. Train ML models for vulnerability assessment
4. Establish behavioral baselines from IoTSAT data
5. Generate remediation strategies

**bva_sat_benchmark.py** will:
1. Load all three datasets
2. Train multiple model variants (standard, deep, attention)
3. Compare performance across datasets
4. Save results to `data/bva_sat_results/bva_sat_comparison_results.csv`
   
> **Note:** Reported results depend on dataset versions, preprocessing, and hardware configuration, and may vary across environments.

### Troubleshooting

If you see "Dataset not found" errors:
- Check that datasets are in the correct folders
- Verify file names match exactly (case-sensitive)
- Use `--data-dir` to specify the correct path

Example error and fix:
```
FileNotFoundError: No UNSW-NB15_*.csv files found
Fix: Ensure UNSW-NB15_1.csv through UNSW-NB15_4.csv are in data/UNSW-NB15/
```

##  Testbed Configuration

The `inventory.yaml` file describes the UNSW-IoTSAT hardware-in-the-loop testbed:

- **Ground Station (GS-01)**: Ubuntu 22.04, BladeRF xA4 SDR
- **Satellite Nodes (SAT-01, SAT-02)**: Raspberry Pi 4, BladeRF x40, multiple sensors
- **Communication Links**: 915-917 MHz, BPSK modulation, CCSDS protocol

##  Framework Architecture
```
Phase 1: Asset Discovery
    ↓ (inventory enumeration)
Phase 2: Vulnerability Scanning  
    ↓ (CVE database search with NVD integration)
Phase 3: Business Assessment
    ↓ (ML-based impact classification)
Phase 4: CVE-to-Attack Mapping
    ↓ (CWE to attack category mapping)
Phase 5: Behavioral Analysis
    ↓ (Statistical anomaly detection)
Output: Comprehensive Security Assessment
```

## Citation

If you use BVA-SAT in your research, The citation will be added once the paper is accepted.


## Requirements

- Python 3.9+ (recommended: 3.10 or 3.11)
- TensorFlow 2.12+
- See `requirements.txt` for complete list

## License

MIT License - See LICENSE file for details

## Authors

- **Osama Abdelhameed** - Framework Design & Implementation
- **Nickolaos Koroniotis** - Supervision & Validation
- **Benjamin Turnbull** - Supervision & Validation

## Contact

For questions or collaboration:
- Email: [o.abdelhameed@unsw.edu.au]
- Issues: [GitHub Issues](https://github.com/Osama-Abdelhameed/BVA-SAT-FRAMEWORK/issues)

## Security Note

This framework is for research and educational purposes. Users are responsible for compliance with applicable laws and regulations when testing security systems.

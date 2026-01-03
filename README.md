# BVA-SAT: Behavioral Vulnerability Assessment Framework for IoT-Based Satellite Security

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)](https://www.tensorflow.org/)

##  Overview

BVA-SAT is a comprehensive five-phase framework that integrates CVE-based vulnerability assessment with behavioral anomaly detection for IoT-enabled satellite systems. This framework addresses the unique security challenges of satellite environments by combining traditional cybersecurity approaches with satellite-specific telemetry analysis.

##  Key Features

- **5-Phase Assessment Pipeline**: Asset Discovery → Vulnerability Scanning → Business Assessment → CVE-to-Attack Mapping → Behavioral Analysis
- **ML-Based Detection**: Three neural network architectures (MLP-Standard, MLP-Deep, Attention-Based)
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

## Usage

### Running the Complete Framework
```python
# Import the framework
from bva_sat_implementation import BVASATFramework

# Initialize the framework
framework = BVASATFramework()

# Phase 1: Asset Discovery
assets = framework.phase_1_asset_discovery('inventory.yaml')
print(f"Discovered {len(assets)} assets")

# Phase 2: Vulnerability Scanning
vulnerabilities = framework.phase_2_vulnerability_scanning()
print(f"Found {len(vulnerabilities)} vulnerabilities")

# Phase 3: Business Assessment
risk_assessment = framework.phase_3_business_vulnerability_assessment()

# Phase 4: CVE to Attack Mapping
attack_mapping = framework.phase_4_vulnerability_remediation()

# Phase 5: Behavioral Analysis
behavioral_results = framework.phase_5_behavioral_baseline()
```

### Running Benchmark Evaluation
```python
# Run benchmark on a specific dataset
python bva_sat_benchmark.py

# The script will:
# 1. Load and preprocess the dataset
# 2. Train three model architectures
# 3. Evaluate performance metrics
# 4. Generate confusion matrices and ROC curves
```

## Performance Results

| Model | UNSW-NB15 | CIC-IDS2017 | UNSW-IoTSAT |
|-------|-----------|-------------|-------------|
| MLP-Standard | 98.78% | 98.64% | 99.94% |
| MLP-Deep | 98.78% | 98.54% | 99.94% |
| Attention | 98.78% | **99.29%** | **99.95%** |

## Datasets

The framework supports three datasets:

1. **UNSW-NB15**: 2.54M samples, 49 features, network intrusion benchmark
   - Download: [UNSW Website](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

2. **CIC-IDS2017**: 2.83M samples, 78 features, enterprise/IoT attacks
   - Download: [UNB Website](https://www.unb.ca/cic/datasets/ids-2017.html)

3. **UNSW-IoTSAT**: 0.40M samples, 35 features, satellite telemetry
   - Available: [GitHub][UNSW Website](https://github.com/Osama-Abdelhameed/UNSW-IoTSAT)

## 🛰️ Testbed Configuration

The `inventory.yaml` file describes the UNSW-IoTSAT hardware-in-the-loop testbed:

- **Ground Station (GS-01)**: Ubuntu 22.04, BladeRF xA4 SDR
- **Satellite Nodes (SAT-01, SAT-02)**: Raspberry Pi 4, BladeRF x40, multiple sensors
- **Communication Links**: 915-917 MHz, BPSK modulation, CCSDS protocol

## 📐 Framework Architecture
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

If you use BVA-SAT in your research, we will provide the paper citation once be accepted.


## Requirements

- Python 3.8+
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

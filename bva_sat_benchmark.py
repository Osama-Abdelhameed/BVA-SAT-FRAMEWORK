"""
BVA-SAT Benchmark Comparison
Comparing BVA-SAT performance against other datasets and approaches

"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import argparse
from pathlib import Path


def resolve_base_dir(cli_value: str | None = None) -> Path:
    """
    Resolve base directory for datasets and outputs.
    Priority:
      1) --data-dir CLI argument
      2) BVA_SAT_DATA_DIR environment variable
      3) directory of this script
    """
    if cli_value:
        return Path(cli_value).expanduser().resolve()

    env_value = os.getenv("BVA_SAT_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()

    return Path(__file__).resolve().parent


class BenchmarkComparison:
    """
    Class for comparing BVA-SAT against benchmark datasets and published results
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.datasets = {}
        self.results = {}
    
    def load_real_unsw_nb15(self, folder_path):
        """
        Load UNSW-NB15 from the four raw CSV splits (UNSW-NB15_1..4.csv)
        using the official feature-name file NUSW-NB15_features.csv.
        
        - Uses 'Label' as binary label (0=normal, 1=attack)
        - Drops attack_cat and non-numeric/categorical fields
        """
        
        print(f"Loading UNSW-NB15 from raw splits in: {folder_path}")
        
        # 1) Load feature names from the metadata file
        
        feat_meta_path = os.path.join(folder_path, "NUSW-NB15_features.csv")
        feat_meta = pd.read_csv(feat_meta_path, encoding="latin1")
        col_names = feat_meta["Name"].tolist()  # 49 names: srcip ... attack_cat, Label
        
        # 2) Load all UNSW-NB15_*.csv (1–4), skip LIST_EVENTS
        csv_files = glob.glob(os.path.join(folder_path, "UNSW-NB15_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No UNSW-NB15_*.csv files found in {folder_path}")
        
        df_list = []
        for f in csv_files:
            base = os.path.basename(f)
            if "LIST_EVENTS" in base:
                print(f"  - Skipping metadata file: {base}")
                continue
                
            print(f"  - Reading raw file: {base}")
            df_part = pd.read_csv(
               f,
               header=None,        # no header in raw files
               encoding="latin1",
               low_memory=False
            )
            df_list.append(df_part)
            
        df = pd.concat(df_list, ignore_index=True)
        
        # 3) Assign proper column names
        if len(col_names) != df.shape[1]:
            raise ValueError(
                f"Feature name count ({len(col_names)}) does not match "
                f"UNSW-NB15 columns ({df.shape[1]})."
            )
            
        df.columns = col_names
        
        # 4) Extract label
        
        if "Label" not in df.columns:
            raise ValueError("UNSW-NB15: 'Label' column not found after assigning names.")
        
        y = df["Label"].astype(int).values  # 0 = normal, 1 = attack
        
        # 5) Drop non-feature columns / text fields to avoid leakage
        
        drop_cols = [
            "Label",
            "attack_cat",   # multi-class text
            "srcip",
            "dstip",
            "proto",
            "state",
            "service",
        ]
        
        X_df = df.drop(columns=drop_cols, errors="ignore")
        
        # Keep numeric only
        X_df = X_df.select_dtypes(include=[np.number])
        
        print(f"UNSW-NB15 samples: {len(df)}, numeric features: {X_df.shape[1]}")
        
        X = X_df.values
        return X, y


    def load_real_cic_ml(self, folder_path):
        """
        Load CIC MachineLearningCVE CSVs (CIC-IDS2017).
        
        - Converts BENIGN -> 0, everything else -> 1
        - Keeps only numeric features
        """
        
        print(f"Loading CIC MachineLearningCVE from: {folder_path}")
        
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {folder_path}")
        df_list = []
        for f in csv_files:
            print(f"  - Reading {os.path.basename(f)}")
            df_list.append(pd.read_csv(f, encoding="latin-1", low_memory=False))
        
        df = pd.concat(df_list, ignore_index=True)
        print(f"Total CIC samples: {len(df)}")
        print(f"CIC columns (first 10): {list(df.columns[:10])}")
        
        # --- robust label column detection ---
        label_col = None
        target_keys = {"label", "attack", "class", "category", "outcome", "result"}
        
        for col in df.columns:
            normalized = col.strip().lower().replace(" ", "").replace("_", "")
            if normalized in target_keys:
                label_col = col
                break
                
        if label_col is None:
            raise ValueError(
                "Could not find label column "
                "(even after normalizing names). Please check one CSV header."
            )
        
        print(f"Using '{label_col}' as label column for CIC ML dataset")
        
        # BENIGN -> 0, everything else -> 1
        y_raw = df[label_col].astype(str)
        y = (y_raw.str.upper().str.strip() != "BENIGN").astype(int).values
        
        # columns to drop (non-features)
        drop_cols = [label_col]
        for col in [
            "Flow ID", "FlowID", "Src IP", "Dst IP",
            "Timestamp", "Source IP", "Destination IP"
        ]:
            if col in df.columns:
                drop_cols.append(col)
                
        # keep only numeric features
        X_df = df.drop(columns=drop_cols, errors="ignore")
        X_df = X_df.select_dtypes(include=[np.number])
        
        print(f"CIC ML numeric feature count (before cleaning): {X_df.shape[1]}")
        
        # clean infinities and NaNs
        # replace +inf / -inf with NaN
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        
        # build mask of valid rows (no NaNs after replacement)
        valid_mask = ~X_df.isna().any(axis=1)
        
        removed = (~valid_mask).sum()
        print(f"Removing {removed} rows with inf/NaN")
        
        X_df = X_df[valid_mask]
        y = y[valid_mask.values]   # keep labels aligned
        
        X = X_df.to_numpy(dtype=np.float32)
        print(f"Final shape after cleaning: X={X.shape}, y={y.shape}")
        
        return X, y
        
        
    def load_real_unsw_iotsat(self, folder_path):
        """
        Load  UNSW-IoTSAT (Satellite-IoT) dataset.
        Uses Attack_Flag as binary label: 0 = normal, 1 = attack.
        """
        print(f"Loading UNSW-IoTSAT from: {folder_path}")

        # Read all UNSW_IoTSAT*.csv files
        csv_files = glob.glob(os.path.join(folder_path, "UNSW_IoTSAT*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No UNSW_IoTSAT*.csv files found in {folder_path}")

        df_list = []
        for f in csv_files:
            print(f"  - Reading {os.path.basename(f)}")
            df_list.append(pd.read_csv(f, encoding='latin-1'))

        df = pd.concat(df_list, ignore_index=True)
        print(f"Total UNSW-IoTSAT samples: {len(df)}")

        # Use Attack_Flag as the main binary label
        label_col = "Attack_Flag"
        if label_col not in df.columns:
            raise ValueError(f"UNSW-IoTSAT: '{label_col}' column not found in data.")

        y = df[label_col].astype(int).values  # 0 = normal, 1 = attack

        # Drop label + all attack-description columns (to avoid data leakage)
        drop_cols = [
            "Attack_Flag",
            "Attack_Type",
            "Attack_Subtype",
            "Attack_Severity",
            "Attack_Duration",
            "Attack_Source_Type",
            "Detection_Confidence",
            "Attack_Timeline_ID",
        ]

        # Also drop obvious non-feature text/time columns
        for col in ["Timestamp", "Reception_Time", "Satellite_ID"]:
            if col in df.columns:
                drop_cols.append(col)

        X_df = df.drop(columns=drop_cols, errors="ignore")
        X_df = X_df.select_dtypes(include=[np.number])

        print(f"UNSW-IoTSAT numeric feature count: {X_df.shape[1]}")
        X = X_df.values
        return X, y

    def load_benchmark_datasets(self):
        """
        Load datasets
        """
        print("Loading benchmark datasets for comparison...")

        # Dataset 1: UNSW-NB15
        print("\n1. UNSW-NB15 Dataset (Network Traffic)")
        unsw_folder = self.base_dir / "UNSW-NB15"
        X_unsw, y_unsw = self.load_real_unsw_nb15(unsw_folder)
        self.datasets['UNSW-NB15'] = {
            'X': X_unsw,
            'y': y_unsw,
            'description': 'Modern network traffic dataset (UNSW-NB15 data)',
            'published_accuracy': None,
            'published_f1': None
        }

        # Dataset 2: CIC-IDS2017 via MachineLearningCVE
        print("\n2. CIC-IDS2017 Dataset (MachineLearningCVE)")
        cic_folder = (self.base_dir / "CICDataset" / "MachineLearningCSV" / "MachineLearningCVE").as_posix()
        X_cic, y_cic = self.load_real_cic_ml(cic_folder)
        self.datasets['CIC-IDS2017'] = {
            'X': X_cic,
            'y': y_cic,
            'description': 'Intrusion detection dataset (CIC MachineLearningCVE)',
            'published_accuracy': None,
            'published_f1': None
        }

        # Dataset 3: UNSW-IoTSAT Dataset (renamed from Satellite-IoT)
        print("\n3. UNSW-IoTSAT Dataset")
        sat_folder = (self.base_dir / "UNSW-IoTSAT").as_posix()
        X_sat, y_sat = self.load_real_unsw_iotsat(sat_folder)
        self.datasets['UNSW-IoTSAT'] = {
            'X': X_sat,
            'y': y_sat,
            'description': 'IoT-based satellite vulnerability dataset (UNSW-IoTSAT)',
            'published_accuracy': None,
            'published_f1': None
        }

        return self.datasets

    def build_optimized_model(self, input_dim, model_variant='standard'):
        """
        Build optimized neural network models for each dataset
        """
        if model_variant == 'standard':
            model = keras.Sequential([
                layers.Input(shape=(input_dim,)),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            
        elif model_variant == 'deep':
            model = keras.Sequential([
                layers.Input(shape=(input_dim,)),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            
        elif model_variant == 'attention':
            # Self-attention mechanism for better feature extraction
            inputs = layers.Input(shape=(input_dim,))
            
            # Feature transformation
            x = layers.Dense(128, activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            
            # Self-attention
            attention = layers.Dense(128, activation='softmax')(x)
            x = layers.Multiply()([x, attention])
            
            # Deep layers
            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dense(32, activation='relu')(x)
            outputs = layers.Dense(1, activation='sigmoid')(x)
            
            model = keras.Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def evaluate_on_dataset(self, dataset_name, dataset_info, model_variant='standard'):
        """
        Evaluate BVA-SAT approach on a specific dataset
        """
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name}")
        print(f"Description: {dataset_info['description']}")
        print(f"{'='*60}")
        
        # Prepare data
        X = dataset_info['X'].copy()
        y = dataset_info['y'].copy()
        
        # Clean data: replace inf and NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Additional cleaning: clip extremely large values
        X = np.clip(X, -1e10, 1e10)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build and compile model
        model = self.build_optimized_model(X.shape[1], model_variant)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), 
                    keras.metrics.Recall(), keras.metrics.AUC()]
        )
        
        # Train model
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        # Calculate class weights to handle imbalance
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        print(f"Class distribution - 0: {np.sum(y_train==0)}, 1: {np.sum(y_train==1)}")
        print(f"Class weights: {class_weight_dict}")
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=64,
            callbacks=[early_stop],
            class_weight=class_weight_dict,
            verbose=0
        )
        
        # Evaluate
        test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(
            X_test, y_test, verbose=0
        )
        
        # Calculate F1 score
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
        from sklearn.metrics import f1_score
        test_f1 = f1_score(y_test, y_pred)
        
        # Store results
        results = {
            'accuracy': test_acc,
            'precision': test_prec,
            'recall': test_rec,
            'f1_score': test_f1,
            'auc': test_auc,
            'loss': test_loss
        }
        
        # Print results
        print(f"\nResults for {dataset_name}:")
        print(f"BVA-SAT Accuracy: {test_acc:.4f}")
        print(f"BVA-SAT F1 Score: {test_f1:.4f}")
        
        return results
    
    def run_comprehensive_comparison(self):
        """
        Run comprehensive comparison across all datasets
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE BENCHMARK COMPARISON STUDY")
        print("="*80)
        
        # Load datasets
        self.load_benchmark_datasets()
        
        # Test multiple model variants
        model_variants = ['standard', 'deep', 'attention']
        
        all_results = {}
        
        for variant in model_variants:
            print(f"\n\nTesting Model Variant: {variant.upper()}")
            print("-"*60)
            
            variant_results = {}
            
            for dataset_name, dataset_info in self.datasets.items():
                results = self.evaluate_on_dataset(
                    dataset_name, dataset_info, variant
                )
                variant_results[dataset_name] = results
            
            all_results[variant] = variant_results
        
        return all_results
    
    def generate_comparison_report(self, results):
        """
        Generate comprehensive comparison report
        """
        print("\n" + "="*80)
        print("FINAL COMPARISON REPORT")
        print("="*80)
        
        # Create comparison dataframe
        comparison_data = []
        
        for variant, variant_results in results.items():
            for dataset, metrics in variant_results.items():
                row = {
                    'Model_Variant': variant,
                    'Dataset': dataset,
                    'Accuracy': metrics['accuracy'],
                    'F1_Score': metrics['f1_score'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'AUC': metrics['auc']
                }
                comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Best performing configuration
        print("\nBest Performing Configurations:")
        print("-"*50)
        
        for dataset in self.datasets.keys():
            dataset_results = df_comparison[df_comparison['Dataset'] == dataset]
            best_config = dataset_results.loc[dataset_results['F1_Score'].idxmax()]
            
            print(f"\n{dataset}:")
            print(f"  Best Model: {best_config['Model_Variant']}")
            print(f"  F1 Score: {best_config['F1_Score']:.4f}")
            print(f"  Accuracy: {best_config['Accuracy']:.4f}")
        
        # Overall best model
        best_overall = df_comparison.loc[df_comparison['F1_Score'].idxmax()]
        print("\n" + "="*50)
        print("OVERALL BEST CONFIGURATION:")
        print(f"  Dataset: {best_overall['Dataset']}")
        print(f"  Model: {best_overall['Model_Variant']}")
        print(f"  F1 Score: {best_overall['F1_Score']:.4f}")
        print(f"  Accuracy: {best_overall['Accuracy']:.4f}")
        
        return df_comparison


def main():
    """
    Main execution for benchmark comparison
    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="BVA-SAT Benchmark Comparison (portable paths)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base folder containing datasets (default: env BVA_SAT_DATA_DIR or script folder)."
    )
    args = parser.parse_args()

    base_dir = resolve_base_dir(args.data_dir)  

    # Initialize comparison framework
    benchmark = BenchmarkComparison(base_dir=base_dir)

    # Run comprehensive comparison
    results = benchmark.run_comprehensive_comparison()

    # Generate comparison report
    df_results = benchmark.generate_comparison_report(results)

    # Save results
    out_dir = base_dir / "bva_sat_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bva_sat_comparison_results.csv"
    df_results.to_csv(out_path, index=False)

    print(f"\nResults saved to: {out_path}")
    print("\n" + "="*80)
    print(f"Comparison study complete. Results saved to {out_path.name}")
    print("="*80)

if __name__ == "__main__":
    main()

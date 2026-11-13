# Stroke classification and SHAP

Course assignment using the Stroke Prediction Dataset and SHAP. The supplied teaching scripts are retained alongside the additional analysis scripts. The source repository for the teaching files was not supplied.

## Data

Download `healthcare-dataset-stroke-data.csv` from [Stroke Prediction Dataset by fedesoriano](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), and place it under `data/`. The dataset is provided by its original authors for educational use. The dataset is not included in this repository. The local file used in the reported experiment has SHA-256 `644d473b05d2797006bd94865e4f8bb057f0c721617911613c82c8fcfc707420`.

## Environment and execution

The experiments used Python 3.12.14. Install dependencies with `python -m pip install -r requirements-analysis.txt`. From the repository root:

```bash
python run_eda.py
python force_plot_example.py
python summary_plot_example.py
python compare_imputation.py
python svm_model.py
python kernel_shap_example.py
python kernel_summary_example.py
```

`00_data_exploration.py` and `03_shap.py` are original teaching scripts. The additional scripts save outputs and account for the SHAP API used in this environment.

## Experimental context

The split uses random_state=2021 with 4,088 training and 1,022 test cases. The random forest experiment follows the supplied zero BMI imputation; the SVM uses training-only median imputation and scaling. Both oversample the training set. The held-out random forest and SVM positive recall were 8% and 48% respectively. SVM precision was 9.80%. These are single-split results.

Kernel SHAP global results summarize 50 deliberately selected test records, including 2 positive cases and the local example, against 50 training background samples. The feature columns, preprocessing, reference distributions, and explainer approximation differ between models. SVM probabilities have not been independently calibrated, and internal calibration after oversampling can be affected by repeated rows across folds.

Data provenance and the original GitHub repository are not verified. These results are educational and do not establish future stroke risk or causal effects.

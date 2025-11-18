# Dataset Characteristics

**[Notebook](exploratory_data_analysis.ipynb)**

## Dataset Information

### Dataset Source
- **Dataset Link:** https://challengedata.ens.fr/participants/challenges/157/
- **Dataset Owner/Contact:** French-Chinese electronics company "Valeo"

### Dataset Characteristics
- **Number of Observations:** The dataset consits of 8277 machine vision images.  [Total number of samples/records in your dataset. For time series data, also specify the temporal resolution (e.g., daily, hourly, etc.)]
- **Number of Features:** Overall 3 Type of label. Label of interest (target prediction) has 6 classes. [Total number of features in your dataset]

### Target Variable/Label
- **Label Name:** label [Name of the target variable/column]
- **Label Type:** Classification [Classification/Regression/Clustering/Other]
- **Label Description:** The label represents the fabrication status of the microelectronic feature depicted. In more detail whether a kind of "bridge" is present or not. [What does this label represent? What is the prediction task?]
- **Label Values:** 
1. Missing:
2. GOOD:
3. Lift-off blanc:
4. Short circuit MOS:
5. Lift-off noir
6. Boucle plate: 
[For classification: list of classes and their meanings. For regression: range of values. For other tasks: describe the label structure]
- **Label Distribution:** [Brief description of class balance for classification or value distribution for regression]

### Feature Description
[Provide a brief description of each feature or group of features in your dataset. If you have many features, group them logically and describe each group. Include information about data types, ranges, and what each feature represents.]

**Example format:**
- **Feature 1 (feature_name):** [Description of what this feature represents, data type, and any relevant details]
- **Feature 2 (feature_name):** [Description of what this feature represents, data type, and any relevant details]
- **Feature Group (group_name):** [Description of a group of related features]

## Exploratory Data Analysis

The exploratory data analysis is conducted in the [exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb) notebook, which includes:

- Data loading and initial inspection
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation analysis
- Data visualization and insights
- Data quality assessment

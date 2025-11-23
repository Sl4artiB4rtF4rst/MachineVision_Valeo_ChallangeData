# Dataset Characteristics

**[Notebook](exploratory_data_analysis.ipynb)**

## Dataset Information

### Dataset Source
- **Dataset Link:** https://challengedata.ens.fr/participants/challenges/157/
- **Dataset Owner/Contact:** French-Chinese electronics company "Valeo"

### Dataset Characteristics
- **Number of Observations:** The dataset consits of 8277 machine vision images.  [Total number of samples/records in your dataset. For time series data, also specify the temporal resolution (e.g., daily, hourly, etc.)]
- **Number of Features:** There are 3 features. First is the image data itself and two strings 'Window' and 'lib'. 


### Target Variable/Label
- **Label Name:** Label [Name of the target variable/column]
- **Label Type:** Classification [Classification/Regression/Clustering/Other]
- **Label Description:** The label represents the fabrication status of the microelectronic feature depicted. In more detail whether a kind of "bridge" is present or not. [What does this label represent? What is the prediction task?]
- **Label Values:** 
1. Missing: bridge-like structure (or contacts) is missing from the depicted area.
2. GOOD: Fully functionung microelectronic structure including bridge and all correct conductive film layers 
3. Lift-off blanc: There seems to be a missing layer in the conductive track (indicated by a different lightness and surface structure). The bridge-like feature is present. Some datapoints seem to be incorrectly labeled as 'Lift-off blanc' although they seem to be 'GOOD'.
4. Short circuit MOS: From visual inspection only there seems no difference between this failure mode and the 'GOOD' areas. Most likely not suitable for training on image data only. 
5. Lift-off noir: looks almost identical to 'Lift-off-blanc' (visual inspection). Some data points also seem to be labelled incorrectly. Needs to be tested whether a model will be able to descern this label at all. 
6. Boucle plate: Some of the bridge-like features for this Label type have a different surface structure. Investigate whether this is correlated to Die or Window features. 
[For classification: list of classes and their meanings. For regression: range of values. For other tasks: describe the label structure]
- **Label Distribution:** [Brief description of class balance for classification or value distribution for regression]

The labels are distributed very unevenly, see plot in exploratory data analysis.
There are approximately 

### Feature Description
[Provide a brief description of each feature or group of features in your dataset. If you have many features, group them logically and describe each group. Include information about data types, ranges, and what each feature represents.]

**Example format:**
- **Feature 1 (Image):** Greyscale 8 bit images taken from a top-down view. Most likely 'normal' camera or scaning electron microscope. Hard to say because there is no scale given for the images.[Description of what this feature represents, data type, and any relevant details]
- **Feature 2 (Window):** 'Window' (most likely) gives the year of manufacture or inspection with the two values 2003 and 2005. [Description of what this feature represents, data type, and any relevant details]
- **Feature 3 (Die):** Lib specifies the 'Die' (Die is a small piece of a silicon wafer on which normally one microelectronic device is situated). There are 4 different Dies numbered 'Die01' to 'Die04'. The different die types can readily be distinguished visually from each other. [Total number of features in your dataset]
[Description of a group of related features]

## Exploratory Data Analysis

The exploratory data analysis is conducted in the [exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb) notebook, which includes:

- Data loading and initial inspection
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation analysis
- Data visualization and insights
- Data quality assessment

In summary the exploratory data analysis shows that 

# Dataset Characteristics

**[Notebook](exploratory_data_analysis.ipynb)**

## Dataset Information

### Dataset Source
- **Dataset Link:** https://challengedata.ens.fr/participants/challenges/157/
- **Dataset Owner/Contact:** French-Chinese electronics company "Valeo"

### Dataset Characteristics
- **Number of Observations:** The dataset consits of 8277 machine vision images.  
- **Number of Features:** There are 3 features. First is the image data itself and two strings 'Window' and 'lib'. 


### Target Variable/Label
- **Label Name:** Label 
- **Label Type:** Classification 
- **Label Description:** The label represents the fabrication status of the microelectronic feature depicted. In more detail whether a kind of "bridge" is present or not. [What does this label represent? What is the prediction task?]
- **Label Values:** 
1. Missing: bridge-like structure (or contacts) is missing from the depicted area.
2. GOOD: Fully functionung microelectronic structure including bridge and all correct conductive film layers 
3. Lift-off blanc: There seems to be a missing layer in the conductive track (indicated by a different lightness and surface structure). The bridge-like feature is present. Some datapoints seem to be incorrectly labeled as 'Lift-off blanc' although they seem to be 'GOOD'.
4. Short circuit MOS: From visual inspection only there seems no difference between this failure mode and the 'GOOD' areas. Most likely not suitable for training on image data only. 
5. Lift-off noir: looks almost identical to 'Lift-off-blanc' (visual inspection). Some data points also seem to be labelled incorrectly. Needs to be tested whether a model will be able to descern this label at all. 
6. Boucle plate: Some of the bridge-like features for this Label type have a different surface structure. Investigate whether this is correlated to Die or Window features. 

- **Label Distribution:** 

The label classes are distributed very unevenly, see plot in exploratory data analysis.
There are approximately 6500 images labelled 'Missing' while only ~1200 with the 'GOOD' label. The other "faulty" label categories have less than 500 datapoints each.  

### Feature Description

**Example format:**
- **Feature 1 (Image):** Greyscale 8 bit images taken from a top-down view. Most likely 'normal' camera or scaning electron microscope. Hard to say because there is no scale given for the images.
- **Feature 2 (Window):** 'Window' (most likely) gives the year of manufacture or inspection with the two values 2003 and 2005. 
- **Feature 3 (Die):** Lib specifies the 'Die' (Die is a small piece of a silicon wafer on which normally one microelectronic device is situated). There are 4 different Dies numbered 'Die01' to 'Die04'. The different die types can readily be distinguished visually from each other. [Total number of features in your dataset]

## Exploratory Data Analysis

The exploratory data analysis is conducted in the [exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb) notebook, which includes:

- Data loading and initial inspection
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation analysis
- Data visualization and insights
- Data quality assessment

In summary the exploratory data analysis shows that there are sufficient data points and good quality, fairly high-resolution images. There are neither missing values in the label data nor corrupt images. 
Every image file name can successfully be mapped to an entry in the label/feature table. 
The Labels are very unevenly distributed. Therefore careful data selection or grouping of labels needs to be done. 
Furthermore the two additional features 'window' and 'Lib' might help in Classification by using them as additional input. How exactly this will be implemented needs to be tested and planned. 
We are positive a high accuracy Classification model (or multi-step models) are in principle possible to construct with the data at hand. 


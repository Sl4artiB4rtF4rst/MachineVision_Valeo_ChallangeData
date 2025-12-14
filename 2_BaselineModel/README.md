# Baseline Model

**[Notebook](baseline_model.ipynb)**

## Baseline Model Results

### Model Selection
- **Baseline Model Type:** Simple CNN Model with 1 convolutional layer and 1 dense layer. No data augmentation was used. The model was trained on the data "as is", which means that there is strong class imbalance. Images were downsampled to a size of 128 x 128 pixels.
- **Rationale:** We have chosen a very basic CNN as the base model. Unlike time series or regression-like tasks for image classification there are no "readily implemented" or easily available non-NN machine learning models available. Or to be more precise: if we do not want to du "manual" image precossing for feature extraction etc., which could be used to for decision tree or random forest approaches, we only have neural networks left. Therefore, we chose a simple CNN with only 1 convolutional layer and 1 dense layer as a baseline model. Such a simple model can already perform the relevant classification in principle while still having enough room for improvement.

### Model Performance
- **Evaluation Metric:** F1-Score (and classification_report) 
- **Performance Score:** 0.988 F1-Score (over all classes) , 0.61 F1-Score (worst class)
- **Classification Report:** 
                     precision    recall  f1-score   support

             0_GOOD       0.99      0.99      0.99       238
        1_Flat loop       0.88      0.47      0.61        15
   2_White lift-off       0.87      0.96      0.92        57
   3_Black lift-off       0.93      1.00      0.96        13
          4_Missing       1.00      1.00      1.00      1316
5_Short circuit MOS       0.87      0.76      0.81        17

           accuracy                           0.99      1656
          macro avg       0.92      0.86      0.88      1656
       weighted avg       0.99      0.99      0.99      1656
	   
Although the model is very simple and containts only few parameters in comparison it alredy gives a higher than expected performance. 

### Evaluation Methodology
- **Data Split:** Train/Validation/Test split:  0.64/0.16/0.2
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score. All metrics cover a distinct subset of classification Performance. Since we have 6 or 7 different labels with different meanings in terms of the data context (microelectronic part inspection) we need to evaluate different relevant metrics to best asses model performance.

We also consider the performance of the worst performing class closely as we want a model that classifies all classes well.

### Metric Practical Relevance

The context of our data and project / challange is (visual) automated inspection of microelectronic parts. The parts are classified to be working / good or defective (with different failure classes) and one class for any data that does not belong to our dataset. 
From a business/company standpoint it is more desireable to exclude a higher number of parts from shipping then to ship faulty parts. Therefore false negatives (in relation to defective parts) should be avoided / penalized higher. 
Depending on the label class we should therefore either use recall, for defect and drift labels, or precision when looking at the good label. As a trade-off for our base model we look at the F1-Score, which is the harmonic mean of precision and recall instead. 

## Next Steps
This baseline model serves as a reference point for evaluating more sophisticated models in the [Model Definition and Evaluation](../3_Model/README.md) phase.
Our next steps are both refining the model - using a more complex NN architecture, hyperparameter tuning - as well as tackling the task of correctly classifying the drift-type datapoints. 
In order to achieve this we plan on either using an approach that is not machine learning based, and therefore does not require correctly labeled data, or labeling a subset of the publicly availabel test dataset. 
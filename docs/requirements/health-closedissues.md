# Client Requirements: Model-Configuration Health Analytics

**Dataset:** `Health-ClosedIssues0000.csv`  
**Client:** Machine-learning platform maintainer

This MOOT file contains model-quality objectives (`MRE-`, `ACC+`, `PRED40+`)
and configuration fields such as estimator count, criterion, leaf size,
impurity decrease, and maximum depth. It does not contain issue timelines or
issue-closure events. The requirements therefore use the dataset's actual
model-configuration domain.

| # | Customer ask | Skill |
|---:|---|---|
| 1 | Summarize model-quality metrics and configuration settings. | remember |
| 2 | Find configurations with unusually strong prediction quality. | remember |
| 3 | Predict model error and accuracy before a full evaluation. | guess |
| 4 | Predict whether a configuration will meet the PRED40 target. | guess |
| 5 | Tell us whether an apparent quality improvement is meaningful or noise. | certify |
| 6 | Tell us whether one configuration family is reliably better than another. | certify |
| 7 | Evaluate configurations as validation results arrive one at a time. | flow |
| 8 | Stop evaluation when the model-quality estimate is stable enough to act. | flow |
| 9 | Explain which hyperparameters are associated with high model error. | blame |
| 10 | Explain which settings are associated with strong accuracy. | blame |
| 11 | Justify a recommended configuration to the platform team. | justify |
| 12 | Explain the trade-off between error, accuracy, and PRED40 performance. | justify |
| 13 | Choose a configuration that minimizes error under a compute budget. | choose |
| 14 | Choose a configuration that balances accuracy and prediction quality. | choose |
| 15 | Suggest the smallest hyperparameter change that improves error. | fix |
| 16 | Recommend a low-cost repair for a configuration with poor accuracy. | fix |
| 17 | Spot an unusual model-quality result or possible measurement error. | spot |
| 18 | Detect a configuration whose behavior is unlike similar settings. | spot |
| 19 | Compare our configuration selector against a historical baseline. | race |
| 20 | Determine whether the selector wins across repeated validation samples. | race |

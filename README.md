# ML_Projects
# Emotion Classification using LLMs

## Overview
This project explores various approaches to classify emotions in text using the GoEmotions dataset. The methods used include:
- Traditional machine learning models such as Naive Bayes, Softmax Regression, and Random Forest.
- Fine-tuning pre-trained BERT models to improve accuracy.

## Dataset
- **GoEmotions Dataset:** 58,000 Reddit comments labeled into 27 emotion categories plus Neutral.
- **Preprocessing:**
  - Filtered out multi-label samples.
  - Split data into training (36,308), validation (4,548), and test (4,590) sets.
  - TF-IDF vectorization for traditional models and BERT tokenization for deep learning models.

## Models Implemented
1. **Naive Bayes Classifier:**
   - Achieved 40.33% test accuracy.
   - Used Laplace smoothing and logarithmic probabilities.
   
2. **Fine-tuned BERT Model:**
   - Unfrozen last two transformer layers for fine-tuning.
   - Achieved 60.98% test accuracy.
   - Optimized with AdamW optimizer.

3. **Baseline Models:**
   - Softmax Regression: 55.97% test accuracy.
   - Random Forest: 53.21% test accuracy.
   - XGBoost: 56.06% test accuracy.

## Key Findings
- Fine-tuning BERT achieved the highest accuracy but required more computational resources.
- Traditional models like Naive Bayes provided efficient baselines.
- Weighted loss functions and oversampling had mixed results.

---

# Neural Networks for Medical Image Classification

## Overview
This project compares different neural network architectures for classifying medical images using the OrganAMNIST dataset. Techniques include:
- Multilayer Perceptrons (MLPs)
- Convolutional Neural Networks (CNNs)
- Pre-trained models (LeNet-5)

## Dataset
- **OrganAMNIST Dataset:** Grayscale organ images categorized into 11 classes.
- **Preprocessing:**
  - Normalized images.
  - Used both 28x28 and 128x128 resolutions.

## Models Implemented
1. **MLPs:**
   - Depth comparison (0, 1, and 2 hidden layers).
   - Best accuracy: 73.42% with two hidden layers.

2. **CNNs:**
   - Achieved 85.63% accuracy on 28x28 images.
   - Performance improved with higher resolutions but required longer training times.

3. **Pre-trained LeNet-5:**
   - Performed better than MLPs but slightly worse than CNNs.

## Key Findings
- CNNs significantly outperformed MLPs due to their ability to capture spatial features.
- Regularization techniques like L1 and L2 improved model generalization.
- Increasing image resolution led to minor improvements but higher risk of overfitting.

---

# Regularization and Model Evaluation in Linear Regression

## Overview
This project explores regularization techniques and model evaluation strategies in linear regression.

## Key Tasks
1. **Linear Regression with Non-Linear Basis Functions:**
   - Used Gaussian basis functions.
   - Optimal number of basis functions determined via SSE.

2. **Bias-Variance Trade-off Analysis:**
   - Observed overfitting and underfitting behaviors with different complexities.

3. **Regularization Techniques:**
   - L1 (Lasso) and L2 (Ridge) regularization applied.
   - Optimal lambda values selected using cross-validation.

4. **Effect of Regularization on Loss:**
   - Contour plots analyzed optimization paths.
   - L1 promoted sparsity, while L2 penalized large weights.

## Key Findings
- Cross-validation is crucial for selecting the best regularization parameters.
- L1 regularization is effective for feature selection.
- L2 regularization helps with generalization by penalizing large coefficients.

---

# Comparative Analysis of Linear and Logistic Regression

## Overview
This project compares linear and logistic regression models using gradient descent and mini-batch optimization techniques.

## Datasets
1. **Infrared Thermography Dataset (Regression Task):** Predicting oral temperature based on environmental factors.
2. **CDC Diabetes Health Indicators Dataset (Classification Task):** Identifying diabetic individuals.

## Key Experiments
1. **Linear Regression:**
   - Compared full-batch vs. mini-batch gradient descent.
   - Evaluated performance with varying batch sizes and learning rates.

2. **Logistic Regression:**
   - Examined accuracy and convergence with different hyperparameters.
   - Addressed class imbalance using undersampling.

## Key Findings
- Smaller batch sizes provided a good trade-off between speed and accuracy.
- Logistic regression performed well with proper class balancing.
- Analytical solutions are suitable for small datasets, while SGD works better for larger ones.

---

## Contributors
- **Saif Al-Alami**
- **Rashid Abu Safia**
- **Zhiheng Zhou**

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/saifalalami/ml-projects.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Jupyter Notebooks for each project.

## Contact
For any inquiries, please contact [Saif Al-Alami](mailto:saif.alalami@example.com).




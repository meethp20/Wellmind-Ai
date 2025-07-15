# WellMind AI: AI-Powered Mental Health Diagnosis

## Description
WellMind AI is a Streamlit web application that leverages machine learning to predict mental health disorders based on user-inputted symptoms and behaviors. The app provides an accessible, interactive interface for both professionals and individuals to gain insights into potential mental health conditions using data-driven models.

## Features
- Predicts mental health disorders from user symptoms
- Compares multiple machine learning models (Logistic Regression, Decision Tree, Random Forest, SVM, KNN)
- Displays model performance metrics and confusion matrices
- Interactive, user-friendly Streamlit interface
- Visualizes dataset statistics and class distributions

## Installation
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Add the dataset:**
   Place `dataset.csv` in the project root or `application/` directory as required.

## Usage
Run the Streamlit app:
```bash
streamlit run app.py
```

- Navigate to `http://localhost:8501` in your browser.
- Use the sidebar to view model performance, make predictions, or explore the dataset.

## Dataset
- The app uses a tab-separated file `dataset.csv` containing anonymized patient symptom data and expert diagnoses.
- **Columns:**
  - Patient Number (dropped during training)
  - 17 symptom/behavior features
  - Expert Diagnose (target)

## Model Information
- The app trains and compares several models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- The best-performing model is used for predictions.

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## License
This project is licensed under the MIT License.

## Contact
For questions or feedback, please contact the maintainer at <your-email@example.com>. 
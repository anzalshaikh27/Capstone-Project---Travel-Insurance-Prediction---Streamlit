# Diabetes Prediction App

## Overview
This project is a web-based application designed to predict the likelihood of diabetes using various health parameters. It uses machine learning algorithms to provide a prediction based on user input. The app is built with Streamlit and uses a RandomForestClassifier for prediction.

## Features
- Interactive sidebar for user input collection.
- Radar chart visualization for comparing health metrics against healthy ranges.
- Feature importance graph to highlight the most significant factors for diabetes prediction.
- Information block with resources and feedback link.
- Customized UI with gradient background and animation effects.
- Downloadable PDF report generation with prediction results.

## How It Works
Users enter their health details, including gender, age, hypertension status, heart disease status, smoking history, BMI, HbA1c level, and blood glucose level. Upon submission, the application processes the inputs through a pre-trained machine learning model to predict whether the user is likely to have diabetes. Results are accompanied by animations and a downloadable PDF report to enhance the user experience.

## Installation
To set up the project locally, follow these steps:

git clone https://github.com/yourusername/Diabetes-Prediction-Capstone.git
cd Diabetes-Prediction-Capstone
pip install -r requirements.txt
streamlit run diabetes_prediction_app.py

css
Copy code

## Usage
Provide a step-by-step guide on how to use the app, possibly with screenshots.

![Sidebar Input](path-to-sidebar-screenshot.png)
*User input via sidebar.*

![Prediction Results](path-to-prediction-screenshot.png)
*Display of prediction results and profile photo.*

![Feature Importance](path-to-feature-importance-screenshot.png)
*Feature importance graph visualized.*

![Download PDF Report](path-to-download-pdf-screenshot.png)
*Button to download the prediction report as a PDF.*

## Resources Section
The 'About & Resources' section provides users with additional information and links to resources on diabetes prevention and healthy living. This aids in user education and empowerment.

![About & Resources](path-to-about-resources-screenshot.png)
*About & Resources block.*

## Feedback
We value your feedback and contributions to improve the app. Please feel free to contact us at your.email@example.com or report an issue on the [GitHub issues page](https://github.com/yourusername/Diabetes-Prediction-Capstone/issues).

## Acknowledgements
Credit to the contributors, data sources, or third-party resources/tools used in the development of this application.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

For a live demonstration, visit the app at [Diabetes Prediction App](https://diabetes-prediction-capstone-run.streamlit.app/)

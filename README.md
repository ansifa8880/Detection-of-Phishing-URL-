# Phishing URL Detection

This project is designed to detect phishing URLs, leveraging machine learning and heuristic methods. Phishing URLs are used to trick users into entering sensitive information such as usernames, passwords, and financial data. This tool helps identify whether a URL is likely to be malicious or safe.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)

## Introduction

Phishing attacks are a serious threat to online security. By analyzing the structure and characteristics of URLs, this project detects whether a given URL is a phishing attempt or a legitimate website. It uses machine learning techniques and heuristic methods for URL classification.

## Features

- **URL Feature Extraction**: The system examines components such as domain names, URL length, use of special characters, and presence of suspicious keywords.
- **Machine Learning Models**: A variety of machine learning models (e.g., Logistic Regression, Random Forest, SVM) are trained to classify URLs as phishing or legitimate.
- **Heuristic Analysis**: Applies rule-based checks on URL patterns and keywords to further identify phishing attempts.
- **Real-time Detection**: Classifies new URLs dynamically as either "Phishing" or "Legitimate" based on the trained model and rule-based system.
 ## Key Features:
URL Feature Extraction: Analyzes components like domain name, length, special characters, etc.
Machine Learning Models: Trained on a dataset of known phishing and legitimate URLs, using models like Logistic Regression, Random Forest, and Support Vector Machine (SVM).
Heuristic Rules: Applies rule-based checks (e.g., presence of suspicious keywords in the URL).
Real-time Detection: Classifies new URLs as "Phishing" or "Legitimate" based on the trained model.

## Technologies Used:
-Python
-Machine Learning (Scikit-learn, TensorFlow)
-Feature Engineering
-Data Preprocessing


## Technologies Used

- **Programming Language**: Python
- **Machine Learning Libraries**: Scikit-learn, TensorFlow (optional for deep learning models)
- **Data Handling**: Pandas, NumPy
- **Web Scraping & URL Analysis**: BeautifulSoup (optional for scraping data)
- **Visualization**: Matplotlib, Seaborn (for data visualization)

## Getting Started

### Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/ansifa8880/phishing-url-detection.git
   ```
2. Navigate into the project directory:
   ```bash
   cd phishing-url-detection
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

After setting up the environment, you can run the phishing detection script:

```bash
python detect_phishing.py
```

The script will classify URLs as **Phishing** or **Legitimate** based on the machine learning model.

For custom URL detection, you can modify the script or pass URLs as command-line arguments (if implemented).

Example usage:

```bash
python detect_phishing.py --url "http://example.com"
```

### Running the Model

Once the model is trained, it can be used to classify new URLs. The `train_model.py` script can be used to train the model on a labeled dataset of phishing and legitimate URLs.

```bash
python train_model.py
```

### Example Output

For each URL, the model will output a classification result:

```
URL: http://example.com
Prediction: Legitimate
```



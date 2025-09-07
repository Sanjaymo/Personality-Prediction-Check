# 🧠 Personality Prediction 

A machine learning-powered system that predicts personality types (Introvert, Extrovert, Ambivert) based on behavioral patterns and social habits using advanced classification algorithms.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Overview

This project leverages machine learning to analyze human behavioral patterns and predict personality types with high accuracy. By examining key behavioral indicators such as social interaction preferences, energy patterns, and communication styles, the system provides insights into personality classification.

### Key Features

- **Advanced ML Model**: Random Forest Classifier for robust personality prediction
- **Comprehensive Analysis**: Multi-metric evaluation (Accuracy, Precision, Recall, F1-score)
- **Interactive Interface**: User-friendly input system for real-time predictions
- **Data Visualization**: Confusion matrix and performance metrics visualization
- **Batch Processing**: Support for single or multiple sample predictions
- **Automated Preprocessing**: Intelligent categorical variable encoding

## 📊 Personality Types

The system classifies individuals into three primary personality categories:

| Type | Characteristics |
|------|----------------|
| **Introvert** | Prefers solitude, smaller social circles, gets energy from alone time |
| **Extrovert** | Thrives in social settings, larger friend groups, energized by social interaction |
| **Ambivert** | Balanced traits, adaptable to various social situations |

## 🔧 Input Parameters

The model analyzes the following behavioral indicators:

1. **Time Spent Alone** (hours/day) - Daily solitude preferences
2. **Stage Fear** (Yes/No) - Public speaking comfort level
3. **Social Event Attendance** (times/month) - Social engagement frequency
4. **Going Outside** (times/week) - Outdoor activity patterns
5. **Energy After Socializing** (Drained/Energized) - Post-social interaction energy levels
6. **Friends Circle Size** (number) - Social network magnitude
7. **Social Media Posting** (posts/week) - Digital communication frequency

## 📁 Project Structure

```
personality-prediction/
│
├── personality_dataset.csv        # Training dataset
│       
├── personality_predictor.py    # Main prediction script
|
├── visualizations/                 # Generated plots and charts
├── requirements.txt               # Project dependencies
├── README.md                     # Project documentation
├── LICENSE                       # License information
└── .gitignore                   # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/personality-prediction.git
   cd personality-prediction
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

Run the prediction system:

```bash
python src/personality_predictor.py
```

Follow the interactive prompts to input behavioral data and receive personality predictions.

## 💡 Example Usage

```
🧠 Personality Prediction System
================================

Enter number of samples to predict: 1

--- Sample 1 Details ---
Time spent alone (hours/day): 4
Stage fear (1=Yes, 2=No): 1
Social events per month: 2
Going outside (times/week): 3
Drained after socializing (1=Yes, 2=No): 2
Friends circle size: 8
Social media posts per week: 5

🎯 Prediction Results
====================
Sample 1: Ambivert (Confidence: 87.3%)
```

## 📈 Model Performance

Our Random Forest Classifier achieves excellent performance metrics:

- **Accuracy**: 91.5%
- **Precision**: 91.7%
- **Recall**: 90.2%
- **F1-Score**: 90.9%

Performance visualizations include:
- Confusion Matrix
- Feature Importance Plot
- ROC Curves
- Precision-Recall Curves

## 🛠️ Technical Details

### Algorithm
- **Model**: Random Forest Classifier
- **Features**: 7 behavioral indicators
- **Preprocessing**: Label encoding for categorical variables
- **Validation**: Cross-validation with stratified sampling

### Dependencies

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## 📚 Dataset

The training dataset contains behavioral patterns from diverse individuals, ensuring robust model generalization across different personality types and demographic groups.

**Note**: The dataset used for training is not included in this repository due to privacy considerations. You can create your own dataset following the input parameter structure.

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📋 Future Enhancements

- [ ] Web-based interface using Flask/Django
- [ ] Real-time data collection integration
- [ ] Mobile application development
- [ ] Advanced feature engineering
- [ ] Deep learning model implementation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Sanjay Choudhari - *Initial work* - https://github.com/Sanjaymo

## 🙏 Acknowledgments

- Scikit-learn community for excellent ML tools
- Contributors to personality psychology research
- Open-source data science community

## 📞 Contact

For questions, suggestions, or collaborations:

- Email: sanjaychoudhari288@gmail.com
- LinkedIn: https://www.linkedin.com/in/sanjay-choudhari-2604a828a/
- Project Link: [https://github.com/Sanjaymo/personality-prediction]

---

⭐ **Star this repository if you find it helpful!** ⭐

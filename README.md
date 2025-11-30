# Machine Learning-Based Dropout Risk Assessment Platform

A comprehensive machine learning-powered platform for identifying at-risk students and managing intervention cases in low-resource educational environments. Built for St. Elizabeth Primary School, Mukuru, Kenya.

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/blswXyO9)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=20146156&assignment_repo_type=AssignmentRepo)

## 🎯 Project Overview

This platform uses a hybrid assessment approach combining rule-based heuristics with Random Forest machine learning models to predict student dropout risk, classify priority levels, and identify specific intervention needs. The system processes unstructured case descriptions from social worker interviews and generates actionable recommendations for early intervention.

## ✨ Key Features

- **Hybrid Risk Assessment**: Combines heuristics (primary) and ML (secondary) for reliable predictions
- **Automated Feature Extraction**: Extracts 82 features from unstructured text (demographics, flags, embeddings)
- **Multi-Task Prediction**: 
  - Priority classification (High/Medium/Low) - 92.1% test accuracy
  - Dropout risk prediction - 99.2% test accuracy, 99.8% AUC-ROC
  - Needs assessment (7 categories) - 96.3% average F1-score
- **Case Management Workflow**: Complete lifecycle from case creation to intervention tracking
- **Role-Based Access Control**: Four user roles (Admin, Social Worker, Teacher, Viewer) with granular permissions
- **Personalized Recommendations**: AI-generated intervention suggestions based on detected needs
- **Real-Time Analytics**: Dashboard with risk distribution, case statistics, and performance metrics

## 🛠️ Technology Stack

### Frontend
- **Streamlit** (v1.28+): Interactive web dashboard
- **HTML/CSS**: Custom styling and UI components

### Backend
- **Python 3.13**: Core programming language
- **PostgreSQL 15+**: Relational database for data persistence
- **SQLAlchemy 2.0+**: ORM for database operations

### Machine Learning
- **scikit-learn** (v1.3+): Random Forest classifiers, PCA, preprocessing
- **sentence-transformers** (v2.2+): Semantic text embeddings (`all-MiniLM-L6-v2`)
- **pandas** (v2.0+): Data manipulation and analysis
- **numpy** (v1.24+): Numerical computing
- **imbalanced-learn** (v0.11+): SMOTE/ADASYN for class balancing

### Authentication & Security
- **Supabase**: Authentication service with email verification
- **Role-Based Access Control (RBAC)**: Granular permission system

## 📊 Model Performance

### Priority Classification
- **Test Accuracy**: 92.1%
- **High Priority Recall**: 84.4% (critical for catching at-risk students)
- **High Priority Precision**: 95.0%
- **High Priority F1-Score**: 89.4%

### Dropout Risk Prediction
- **Test Accuracy**: 99.2%
- **Test AUC-ROC**: 99.8% (excellent discrimination)
- **Test Precision**: 97.1%
- **Test Recall**: 97.1%
- **Test F1-Score**: 97.1%

### Needs Assessment (Average across 7 needs)
- **Average Precision**: 100%
- **Average Recall**: 89.3%
- **Average F1-Score**: 96.3%
- **Best Performing**: School Fees (100% F1), Economic (100% F1), Family Support (100% F1), Housing (99.8% F1), Health (93.3% F1), Counseling (91.9% F1), Food (89.2% F1)

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- PostgreSQL 15+
- Supabase account (for authentication)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/victorrndungu/ML-Dropout-Risk-Assessment.git
   cd ML-Dropout-Risk-Assessment
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   - Create `.env` file with Supabase credentials:
     ```
     SUPABASE_URL=your_supabase_url
     SUPABASE_ANON_KEY=your_anon_key
     SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
     ```

5. **Setup database**
   ```bash
   python3 database_setup.py
   ```

6. **Run the application**
   ```bash
   streamlit run app_postgres_exact.py --server.port 8504
   ```

7. **Access the application**
   - Open browser: `http://localhost:8504`
   - Login with your credentials

## 📁 Project Structure

```
ML-Dropout-Risk-Assessment/
├── app_postgres_exact.py          # Main Streamlit application
├── complete_upload_processor.py   # Feature extraction and prediction pipeline
├── build_features.py              # Feature extraction from text
├── ml_training.py                 # Model training scripts
├── heuristics.py                  # Rule-based risk scoring
├── recommendations.py             # Intervention recommendation engine
├── database_setup.py              # Database schema and setup
├── rbac.py                       # Role-based access control
├── models/                        # Trained ML models
├── models_enhanced/               # Ensemble voting models
├── viz_outputs/                   # Performance visualizations and EDA charts
├── usable/                        # Original case data (265 files)
├── usable_aug/                    # Augmented case data (1,060 files)
└── requirements.txt               # Python dependencies
```

## 🔄 System Workflow

1. **Case Input**: Social worker enters unstructured case description
2. **Feature Extraction**: System extracts 82 features (demographics, flags, embeddings)
3. **Hybrid Assessment**: 
   - Heuristics calculates risk score (primary)
   - ML models predict priority, dropout risk, and needs (secondary)
4. **Risk Consolidation**: Results combined with heuristics taking precedence
5. **Recommendation Generation**: Personalized interventions mapped to detected needs
6. **Case Management**: Track interventions, update status, reassess cases

## 👥 User Roles

- **Admin**: Full system access, user management, analytics
- **Social Worker**: Create cases, manage interventions, view analytics
- **Teacher**: Submit case requests, view own requests
- **Viewer**: Read-only access to reports and analytics

## 📈 Dataset

- **Original Cases**: 265 anonymized student case descriptions
- **Augmented Cases**: 1,060 synthetic variants (4x augmentation)
- **Total Dataset**: 1,325 cases
- **Features**: 82 dimensions (6 structured + 12 flags + 64 PCA-reduced embeddings)
- **Labels**: Priority (3-class), Dropout Risk (binary), Needs (7 binary)

## 🧪 Testing

The system has been tested across multiple domains:
- **Functional Testing**: 20 test cases (100% pass rate)
- **ML Prediction Testing**: 10 test cases (100% pass rate)
- **Security Testing**: 8 test cases (SQL injection, XSS, authentication)
- **Database Testing**: 8 test cases (integrity, constraints, performance)
- **UI Testing**: 10 test cases (navigation, forms, responsiveness)
- **Performance Testing**: 5 test cases (sub-5-second analysis, sub-3-second page loads)

## 📚 Documentation

- `PROJECT_DOCUMENTATION.md`: Comprehensive system documentation
- `POSTGRESQL_SETUP.md`: Database setup guide
- `AUTHENTICATION_SETUP.md`: Authentication configuration
- `TECHNICAL_ML_ANALYSIS.md`: Detailed ML model analysis
- `ACTUAL_EXECUTION_FLOW.md`: System execution flow documentation

## 🤝 Contributing

This is a final year project. For questions or collaboration, please contact the repository owner.

## 📄 License

This project is part of a final year Computer Science degree program at Strathmore University, Nairobi, Kenya.

## 👤 Author

**Ndung'u Victor Kahindo**  
Student ID: 150668  
ICS 4B  
Strathmore University

## 🙏 Acknowledgments

- Prof. Vincent Omwenga (Supervisor)
- Mr. Byron Mugesiah (Social Worker, St. Elizabeth Primary School, Mukuru)
- St. Elizabeth Primary School, Mukuru
- Strathmore University

---

**Status**: Production-ready system with full authentication, ML predictions, and PostgreSQL backend.

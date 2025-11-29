# Machine Learning-Based Dropout Risk Assessment Platform

A comprehensive machine learning-powered platform for identifying at-risk students and managing intervention cases in low-resource educational environments. Built for St. Elizabeth Primary School, Mukuru, Kenya.

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/blswXyO9)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=20146156&assignment_repo_type=AssignmentRepo)

## Project Overview

This platform uses a hybrid assessment approach combining rule-based heuristics with Random Forest machine learning models to predict student dropout risk, classify priority levels, and identify specific intervention needs. The system processes unstructured case descriptions from social worker interviews and generates actionable recommendations for early intervention.

## Key Features

- **Hybrid Risk Assessment**: Combines heuristics (primary) and ML (secondary) for reliable predictions
- **Automated Feature Extraction**: Extracts 82 features from unstructured text (demographics, flags, embeddings)
- **Multi-Task Prediction**: 
  - Priority classification (High/Medium/Low) - 90.2% test accuracy
  - Dropout risk prediction - 97.7% test accuracy, 96.7% AUC-ROC
  - Needs assessment (7 categories) - 95.6% average F1-score
- **Case Management Workflow**: Complete lifecycle from case creation to intervention tracking
- **Role-Based Access Control**: Three user roles (Admin, Social Worker, Teacher) with granular permissions
- **Recommendations**: Generated intervention suggestions based on detected needs
- **Real-Time Analytics**: Dashboard with risk distribution, case statistics, and performance metrics

## Screenshots of the System
1. Homepage
<img width="1466" height="710" alt="image" src="https://github.com/user-attachments/assets/f2cecfde-a31f-4598-9229-5a0b1b6dcea1" />
2. Create Case Page
<img width="1466" height="710" alt="image" src="https://github.com/user-attachments/assets/3b2a8c61-8161-4c82-a5fa-024186e16df8" />
After Loading Case.
<img width="1466" height="795" alt="image" src="https://github.com/user-attachments/assets/012be030-dcfb-4902-b4e7-52e69c193172" />
3. Case Management
<img width="1466" height="795" alt="image" src="https://github.com/user-attachments/assets/2d40e682-abc1-43d2-96e2-b090a56a2111" />
Case Profile Menu 
<img width="1466" height="795" alt="image" src="https://github.com/user-attachments/assets/13757e81-7f48-4c89-b0f9-71b8f3f237d3" />
Post Reassessment
<img width="1313" height="764" alt="image" src="https://github.com/user-attachments/assets/7531d041-9f89-4166-a1fd-e44219be903f" />
Assessment History View
<img width="1313" height="764" alt="image" src="https://github.com/user-attachments/assets/f3b6fe33-00b3-4d00-a49c-66ad4299f0f0" />
4. Overview
<img width="1164" height="764" alt="image" src="https://github.com/user-attachments/assets/7ebcad7c-a5bd-4caf-9ae8-7c777983806c" />
<img width="1305" height="469" alt="image" src="https://github.com/user-attachments/assets/1440c6e3-2cd3-4bed-af23-631acb3ef918" />
<img width="1305" height="584" alt="image" src="https://github.com/user-attachments/assets/d20aaf66-b1c6-4e12-8fe6-2316fc56c511" />
<img width="1305" height="266" alt="image" src="https://github.com/user-attachments/assets/5b1111d5-8817-4a01-946b-2cb8216cf96f" />

## Technology Stack

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
- **Test Accuracy**: 90.2%
- **High Priority Recall**: 65.4% (critical for catching at-risk students)
- **High Priority Precision**: 81.0%
- **High Priority F1-Score**: 72.3%

### Dropout Risk Prediction
- **Test Accuracy**: 97.7%
- **Test AUC-ROC**: 96.7% (excellent discrimination)
- **Test Precision**: 82.4%
- **Test Recall**: 82.4%
- **Test F1-Score**: 82.4%

### Needs Assessment (Average across 7 needs)
- **Average Precision**: 100%
- **Average Recall**: 89.3%
- **Average F1-Score**: 93.8%

## Quick Start

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
├── usable/                        # Original case data (265 files)
├── usable_aug/                    # Augmented case data (1,060 files)
└── requirements.txt               # Python dependencies
```

## System Workflow

1. **Case Input**: Social worker enters unstructured case description
2. **Feature Extraction**: System extracts 82 features (demographics, flags, embeddings)
3. **Hybrid Assessment**: 
   - Heuristics calculates risk score (primary)
   - ML models predict priority, dropout risk, and needs (secondary)
4. **Risk Consolidation**: Results combined with heuristics taking precedence
5. **Recommendation Generation**: Personalized interventions mapped to detected needs
6. **Case Management**: Track interventions, update status, reassess cases

## User Roles

- **Admin**: Full system access, user management, analytics
- **Social Worker**: Create cases, manage interventions, view analytics
- **Teacher**: Submit case requests, view own requests

## Dataset

- **Original Cases**: 265 anonymized student case descriptions
- **Augmented Cases**: 1,060 synthetic variants (4x augmentation)
- **Total Dataset**: 1,325 cases
- **Features**: 82 dimensions (6 structured + 12 flags + 64 PCA-reduced embeddings)
- **Labels**: Priority (3-class), Dropout Risk (binary), Needs (7 binary)

## Testing

The system has been tested across multiple domains:
- **Functional Testing**: 20 test cases (100% pass rate)
- **ML Prediction Testing**: 10 test cases (100% pass rate)
- **Security Testing**: 8 test cases (SQL injection, XSS, authentication)
- **Database Testing**: 8 test cases (integrity, constraints, performance)
- **UI Testing**: 10 test cases (navigation, forms, responsiveness)
- **Performance Testing**: 5 test cases (sub-5-second analysis, sub-3-second page loads)


## Contributing

This is a final year project. For questions or collaboration, please contact the repository owner.

## License

This project is part of a final year Computer Science degree program at Strathmore University, Nairobi, Kenya.

## Author

**Ndung'u Victor Kahindo**  
Student ID: 150668  
ICS 4B  
Strathmore University

## Acknowledgments

- Mr. Byron Mugesiah (Social Worker, St. Elizabeth Primary School, Mukuru)
- St. Elizabeth Primary School, Mukuru
- Strathmore University

---

**Status**: Production-ready system with full authentication, ML predictions, and PostgreSQL backend.

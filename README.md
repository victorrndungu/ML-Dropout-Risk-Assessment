# Machine Learning-Based Dropout Risk Assessment Platform

A comprehensive early intervention system for identifying at-risk students and providing data-driven recommendations for social workers, teachers, and administrators. This platform combines rule-based heuristics with machine learning models to assess dropout risk, identify critical needs, and generate actionable intervention recommendations.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [User Roles and Permissions](#user-roles-and-permissions)
- [Screenshots](#screenshots)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

## Overview

This platform addresses the critical challenge of student dropout in vulnerable communities by providing:

- **Automated Risk Assessment**: Machine learning models analyze student profiles to predict dropout risk and priority levels
- **Needs Identification**: Multi-label classification identifies seven critical need categories (food, school fees, housing, economic support, family support, health, counseling)
- **Intervention Recommendations**: AI-powered recommendations based on detected needs and risk levels
- **Case Management**: Complete workflow for tracking cases, interventions, and outcomes
- **Role-Based Dashboards**: Tailored interfaces for social workers, teachers, administrators, and stakeholders

### Key Technologies

- **Frontend**: Streamlit web application
- **Backend**: Python 3.8+
- **Database**: PostgreSQL
- **Authentication**: Supabase Auth
- **Machine Learning**: Scikit-learn (Random Forest, Decision Trees)
- **NLP**: SentenceTransformers for text embeddings
- **Data Processing**: Pandas, NumPy

## System Architecture

The system follows a three-tier architecture:

1. **Presentation Layer**: Streamlit web interface with role-based dashboards
2. **Application Layer**: 
   - Feature extraction and preprocessing
   - Machine learning prediction engine
   - Heuristics-based risk scoring
   - Recommendation generation
   - Case management logic
3. **Data Layer**: PostgreSQL database storing profiles, assessments, cases, and interventions

### Data Flow

1. **Input**: Student case descriptions (text) entered via web interface or uploaded as files
2. **Feature Extraction**: 
   - Structured data extraction (age, class, exam scores, family composition)
   - Keyword detection for 7 critical flags
   - Text embeddings using SentenceTransformer
   - PCA dimensionality reduction
3. **Assessment**:
   - Heuristic scoring (rule-based)
   - ML predictions (Random Forest models)
   - Hybrid decision logic combining both approaches
4. **Output**: Risk assessments, priority levels, needs identification, and intervention recommendations

## Features

### Core Functionality

- **Risk Assessment**: Automated priority classification (High/Medium/Low) and dropout risk prediction
- **Needs Analysis**: Multi-label needs detection across 7 categories
- **Case Management**: Complete CRUD operations for cases, interventions, and status tracking
- **Search and Filtering**: Advanced filtering by status, priority, risk level, and search functionality
- **Analytics Dashboard**: Risk distribution charts, needs breakdown, and trend analysis
- **Intervention Timeline**: Track intervention history and status changes over time

### User-Specific Features

**Social Workers**:
- Create and manage cases
- View risk assessments and recommendations
- Track interventions and outcomes
- Access comprehensive case history
- Generate workload summaries

**Teachers**:
- Submit case monitoring requests
- View assigned cases
- Limited case viewing capabilities

**Administrators**:
- User management and approval
- System-wide analytics
- Model performance metrics
- Administrative controls

**Viewers/Partners**:
- Read-only access to analytics
- Risk distribution overview
- System performance metrics

## Installation

### Prerequisites

- Python 3.8 or higher
- PostgreSQL 12 or higher
- pip package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ML-Dropout-Risk-Assessment.git
cd ML-Dropout-Risk-Assessment
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install PostgreSQL

**macOS (using Homebrew)**:
```bash
brew install postgresql
brew services start postgresql
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
```

**Windows**: Download and install from [PostgreSQL official website](https://www.postgresql.org/download/windows/)

### Step 4: Set Up PostgreSQL Database

1. Create a new database:
```bash
createdb dropout_risk_db
```

2. Create a user (optional, can use default postgres user):
```bash
psql postgres
CREATE USER your_username WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE dropout_risk_db TO your_username;
\q
```

3. Update `db_config.json` with your database credentials:
```json
{
  "host": "localhost",
  "port": "5432",
  "database": "dropout_risk_db",
  "username": "your_username",
  "password": "your_password"
}
```

### Step 5: Set Up Supabase Authentication

1. Create a Supabase project at [supabase.com](https://supabase.com)
2. Get your project URL and anon key from Supabase dashboard
3. Create a `.env` file in the project root:
```
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
```

### Step 6: Initialize Database Schema

```bash
python database_setup.py
```

This will create all necessary tables in your PostgreSQL database.

## Data Setup

**Important**: The training data and case files are not included in this repository due to privacy and size constraints. You need to set up your data before running the system.

### Option 1: Using Existing Training Data

If you have access to the original training data:

1. **Place training data files**:
   - Create a `usable/` directory in the project root
   - Place your `.txt` case files in this directory
   - Each file should contain a student case description

2. **Generate features dataset**:
```bash
python build_features.py
```

This will create `usable/features_dataset.csv` with extracted features.

3. **Train ML models**:
```bash
python ml_training.py
```

This will generate trained models in the `models/` directory:
- `random_forest_needs_model.pkl`
- `random_forest_priority_model.pkl`
- `random_forest_dropout_model.pkl`
- `feature_scaler.pkl`
- `pca_transformer.pkl`
- `priority_encoder.pkl`
- `feature_names.json`

4. **Migrate data to PostgreSQL**:
```bash
python postgres_exact_replica.py
```

### Option 2: Start with Empty Database

If you don't have training data, you can start with an empty database:

1. The system will work with new cases entered through the web interface
2. Models must be trained first (see Option 1) or use pre-trained models if available
3. New cases will be processed and stored in the database

### Option 3: Using Sample Data

To test the system without real data:

1. Create sample case files in `usable/` directory
2. Each file should follow this format:
```
The pupil is a [age]-year-old [gender] in Class [level].
[Case description including family situation, housing conditions, 
economic status, health status, academic performance, etc.]
```

3. Follow steps 2-4 from Option 1

### Data Directory Structure

```
ML-Dropout-Risk-Assessment/
├── usable/              # Original case files (.txt)
│   └── *.txt
├── usable_aug/          # Augmented case files (optional)
│   └── *.txt
├── models/              # Trained ML models
│   ├── random_forest_needs_model.pkl
│   ├── random_forest_priority_model.pkl
│   ├── random_forest_dropout_model.pkl
│   ├── feature_scaler.pkl
│   ├── pca_transformer.pkl
│   ├── priority_encoder.pkl
│   └── feature_names.json
└── case_data/           # Case tracking data (auto-generated)
    ├── case_tracking.csv
    └── interventions_log.csv
```

## Configuration

### Database Configuration

Edit `db_config.json`:
```json
{
  "host": "localhost",
  "port": "5432",
  "database": "dropout_risk_db",
  "username": "your_username",
  "password": "your_password"
}
```

### Environment Variables

Create a `.env` file for sensitive configuration:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
DB_HOST=localhost
DB_PORT=5432
DB_NAME=dropout_risk_db
DB_USER=your_username
DB_PASSWORD=your_password
```

### Model Configuration

Model parameters can be adjusted in `ml_training.py`:
- Random Forest hyperparameters (n_estimators, max_depth, etc.)
- Feature selection
- Train/test split ratio
- Class balancing strategies

## Usage Guide

### Starting the Application

1. **Start PostgreSQL** (if not running as a service):
```bash
# macOS/Linux
pg_ctl -D /usr/local/var/postgres start

# Or if using systemd
sudo systemctl start postgresql
```

2. **Run the Streamlit application**:
```bash
streamlit run app_postgres_exact.py --server.port 8504
```

3. **Access the application**:
Open your browser and navigate to `http://localhost:8504`

### First-Time Setup

1. **Create Admin Account**:
   - Click "Sign Up" on the login page
   - Register with your email and password
   - Note: Admin approval is required for new accounts

2. **Approve Users** (Admin only):
   - Log in as an admin
   - Navigate to "User Management" or "Admin Approval"
   - Approve pending user registrations

3. **Load or Create Data**:
   - If you have training data, follow the Data Setup section
   - Otherwise, start creating cases through the web interface

### Creating a New Case

1. Navigate to "Case Management" > "Create New Case"
2. Fill in student identifiers:
   - Student Name
   - Age
   - Class Level
   - Admission Number (optional)
3. Enter case description in the text area
4. Click "Store & Analyze"
5. Review the analysis results:
   - Risk flags table
   - Priority level
   - Dropout risk assessment
   - Identified needs
   - Recommended interventions
6. Select interventions from the checklist
7. Add custom interventions if needed
8. Click "Add to Case Management" to save

### Managing Cases

1. Navigate to "Case Management" > "Manage Cases"
2. Use filters to find specific cases:
   - Filter by status (new, in_progress, closed, etc.)
   - Filter by priority (high, medium, low)
   - Search by student name or UID
3. View case details by clicking on a case
4. Update case status as interventions progress
5. Add interventions and track outcomes

### Viewing Analytics

1. Navigate to "Dashboard" or "Analytics & Overview"
2. View key metrics:
   - Total students assessed
   - Risk distribution
   - Needs breakdown
   - Case status overview
3. Explore charts and visualizations
4. Export data if needed

### Teacher Request Workflow

1. **Teachers** submit case monitoring requests:
   - Navigate to "Case Management" > "Teacher Requests"
   - Click "Submit New Request"
   - Enter minimal student information (name, age, class, admission number)
   - Add optional teacher notes
   - Submit request

2. **Social Workers** process requests:
   - View pending requests in "Case Management" > "Teacher Requests"
   - Click on a request to review
   - Add full case description
   - Click "Store & Analyze" to assess
   - Review and select interventions
   - Mark request as completed

## User Roles and Permissions

### Social Worker

**Permissions**:
- View all cases
- Create new cases
- Manage cases and interventions
- View analytics and reports
- Re-assess cases
- Process teacher requests

**Accessible Pages**:
- Home
- Case Management (full access)
- Risk Assessment
- Needs Overview
- Analytics & Overview
- Dashboard

### Teacher

**Permissions**:
- Submit case monitoring requests
- View assigned cases only
- Limited case viewing

**Accessible Pages**:
- Home
- Case Management > Teacher Requests (submit only)
- View assigned cases

### Administrator

**Permissions**:
- All social worker permissions
- User management
- Approve new user registrations
- System configuration
- Model metrics access

**Accessible Pages**:
- All pages available to social workers
- User Management
- Admin Approval
- Model Metrics

### Viewer/Partner

**Permissions**:
- Read-only access to analytics
- View risk distribution
- View system overview

**Accessible Pages**:
- Home
- Dashboard (read-only)
- Analytics & Overview (read-only)

## Screenshots

### 1. Login/Authentication Page

![Login Page](screenshots/01-login-page.png)

*Shows the Supabase authentication interface with email/password login and sign-up options. Users can register new accounts or sign in with existing credentials.*

**To add screenshot**: Replace `screenshots/01-login-page.png` with your actual screenshot file path, or use a relative path from the README location.

---

### 2. Homepage - Social Worker

![Social Worker Homepage](screenshots/02-homepage-social-worker.png)

*Shows the welcome message, platform description, benefits section, and "Create a Case to get started" call-to-action button. The homepage is customized based on user role.*

---

### 3. Homepage - Admin

![Admin Homepage](screenshots/03-homepage-admin.png)

*Shows admin-specific content including permissions and access rights information. Admins see additional administrative features and system overview.*

---

### 4. Homepage - Teacher

![Teacher Homepage](screenshots/04-homepage-teacher.png)

*Shows teacher-specific content and available features. Teachers can see their assigned cases and submit new case monitoring requests.*

---

### 5. Case Creation Page

![Case Creation](screenshots/05-case-creation.png)

*Shows the "Create New Case" interface with student identifiers section (name, age, class, admission number), case description text area, and "Store & Analyze" button. This is where social workers input new student cases.*

---

### 6. Case Analysis Results

![Case Analysis Results](screenshots/06-case-analysis-results.png)

*Shows the analysis results after creating a case, including the flags table (7 critical needs indicators), priority level (High/Medium/Low), dropout risk assessment, identified needs, and recommended interventions checklist with categorized tables.*

---

### 7. Case Management - Manage Cases

![Manage Cases](screenshots/07-case-management-manage-cases.png)

*Shows the "Manage Cases" tab with the cases table, filters (status, priority), search functionality, and case details view. Social workers can filter, search, and manage all cases from this interface.*

---

### 8. Case Management - Teacher Requests

![Teacher Requests](screenshots/08-teacher-requests.png)

*Shows the "Teacher Requests" tab with pending teacher requests table and request processing interface. Social workers can view, process, and complete teacher-submitted case monitoring requests.*

---

### 9. Case Profile/Details

![Case Profile](screenshots/09-case-profile.png)

*Shows an individual case profile page with complete case information including assessment history, interventions timeline, status changes, notes, and all related data for a specific student case.*

---

### 10. Needs Overview

![Needs Overview](screenshots/10-needs-overview.png)

*Shows the "Needs Overview" page with cases categorized by need type (food, school fees, housing, economic support, family support, health, counseling) and ordered by risk level within each category.*

---

### 11. Analytics Dashboard

![Analytics Dashboard](screenshots/11-analytics-dashboard.png)

*Shows the main analytics dashboard with risk distribution charts, needs breakdown visualizations, case status overview, and key metrics. Provides system-wide insights and trends.*

---

### 12. Model Metrics

![Model Metrics](screenshots/12-model-metrics.png)

*Shows the "Model Metrics" page with ML model performance metrics including accuracy, precision, recall, F1-scores, confusion matrices, and validation vs test performance comparisons for priority, needs, and dropout risk models.*

---

### 13. Intervention Timeline

![Intervention Timeline](screenshots/13-intervention-timeline.png)

*Shows intervention history and timeline for a case, displaying chronological view of interventions, status changes, follow-up scheduling, and outcomes. Helps track case progress over time.*

---

### 14. User Management (Admin)

![User Management](screenshots/14-user-management.png)

*Shows the admin user management interface with user list, role assignments, approval workflow, and administrative controls. Admins can approve new registrations and manage user roles.*

---

### 15. Batch Processing

![Batch Processing](screenshots/15-batch-processing.png)

*Shows the batch processing interface for bulk case upload, batch analysis results, and export options. Allows processing multiple cases simultaneously for efficiency.*

---

**Note**: To add screenshots, place your image files in the repository (you can create a `screenshots/` folder in the root directory or use any path structure you prefer) and update the image paths above. Use standard Markdown image syntax: `![Alt text](path/to/image.png)`. You can also use absolute URLs if hosting images externally.

## Technical Details

### Machine Learning Models

The system uses three Random Forest classifiers:

1. **Priority Model**: Multi-class classification (High/Medium/Low priority)
2. **Needs Model**: Multi-label classification (7 need categories)
3. **Dropout Risk Model**: Binary classification (at-risk/not at-risk)

**Model Architecture**:
- Base algorithm: Random Forest (100 estimators)
- Feature set: 82 features including:
  - Structured features (age, exam scores, family composition)
  - Binary flags (7 critical needs indicators)
  - Text embeddings (384-dimensional, reduced to 50 via PCA)
  - Composite features

**Training Process**:
- Train/test split: 80/20
- Stratified sampling for balanced classes
- Class weighting for imbalanced datasets
- Cross-validation for hyperparameter tuning

### Feature Engineering

**Structured Features**:
- Age, class level, exam scores
- Meals per day, siblings count
- Text length, sentence count

**Keyword Detection**:
- Housing flags: iron sheets, single room, shared bed, no electricity
- Economic flags: rent arrears, landlord lock, unstable work, no school fees
- Family flags: father absent, single parent, mother hawker
- Health and hunger indicators

**Text Embeddings**:
- Model: `all-MiniLM-L6-v2` (SentenceTransformer)
- Dimension: 384 (reduced to 50 via PCA)
- Captures semantic meaning of case descriptions

### Database Schema

**Core Tables**:
- `profiles`: Student case information
- `profile_embeddings`: Text embeddings for each profile
- `risk_assessments`: ML predictions and heuristic scores
- `case_records`: Case management records
- `interventions`: Intervention tracking
- `case_requests`: Teacher monitoring requests
- `case_assessment_history`: Historical assessments
- `users`: User accounts (Supabase Auth)

### API Endpoints

The system uses Supabase Auth for authentication:
- Sign up: `POST /auth/v1/signup`
- Sign in: `POST /auth/v1/token`
- User session: Managed via Supabase client

### Performance Considerations

- **Model Loading**: Models are cached using `@st.cache_resource` for faster page loads
- **Data Loading**: PostgreSQL queries are optimized with indexes
- **Embedding Generation**: SentenceTransformer model is loaded once and reused
- **Batch Processing**: Cases processed in batches for efficiency

## Troubleshooting

### Common Issues

**Issue: "Error loading models"**
- **Solution**: Ensure `models/` directory contains all required `.pkl` files. Run `python ml_training.py` to generate models.

**Issue: "Database connection failed"**
- **Solution**: 
  - Verify PostgreSQL is running: `pg_isready` or `sudo systemctl status postgresql`
  - Check `db_config.json` credentials
  - Ensure database exists: `psql -l | grep dropout_risk_db`

**Issue: "No data available"**
- **Solution**: 
  - If using existing data, ensure files are in `usable/` directory
  - Run `python build_features.py` to generate features
  - Run `python postgres_exact_replica.py` to migrate to PostgreSQL
  - Or create new cases through the web interface

**Issue: "Supabase authentication error"**
- **Solution**:
  - Verify `.env` file exists with correct `SUPABASE_URL` and `SUPABASE_KEY`
  - Check Supabase project is active
  - Ensure email verification is configured correctly

**Issue: "Port 8504 already in use"**
- **Solution**: 
  - Use a different port: `streamlit run app_postgres_exact.py --server.port 8505`
  - Or kill the process using port 8504: `lsof -ti:8504 | xargs kill`

**Issue: "Module not found" errors**
- **Solution**: 
  - Ensure all dependencies are installed: `pip install -r requirements.txt`
  - Check Python version: `python --version` (should be 3.8+)
  - Verify virtual environment is activated if using one

**Issue: "Permission denied" errors**
- **Solution**:
  - Check user role in Supabase Auth
  - Verify RBAC permissions in `rbac.py`
  - Ensure user account is approved (for non-admin roles)

### Database Issues

**Reset Database**:
```bash
# Drop and recreate database
dropdb dropout_risk_db
createdb dropout_risk_db
python database_setup.py
```

**Check Database Connection**:
```bash
python inspect_postgres_db.py
```

**View Database Tables**:
```bash
psql dropout_risk_db
\dt
\q
```

### Model Issues

**Retrain Models**:
```bash
python ml_training.py
```

**Verify Model Files**:
```bash
ls -lh models/*.pkl
```

**Test Model Loading**:
```python
import joblib
model = joblib.load('models/random_forest_priority_model.pkl')
print("Model loaded successfully")
```

## Development

### Project Structure

```
ML-Dropout-Risk-Assessment/
├── app_postgres_exact.py          # Main Streamlit application
├── database_setup.py               # Database schema and setup
├── postgres_exact_replica.py      # PostgreSQL data operations
├── build_features.py               # Feature extraction
├── ml_training.py                  # ML model training
├── heuristics.py                   # Rule-based risk scoring
├── complete_upload_processor.py    # Prediction pipeline
├── recommendations.py              # Intervention recommendations
├── case_management.py             # Case management logic
├── auth_supabase.py               # Supabase authentication
├── auth_supabase_ui.py            # Auth UI components
├── rbac.py                         # Role-based access control
├── utils.py                        # Utility functions
├── requirements.txt                # Python dependencies
├── db_config.json                  # Database configuration
├── models/                         # Trained ML models
├── usable/                         # Training data (not in repo)
└── docs/                           # Documentation
    └── screenshots/                # Screenshot placeholders
```

### Branch Structure

- `main`: Production-ready code
- `2-authentication`: Authentication system
- `4-core-crud-data-intake`: Database and CRUD operations
- `5-preparation-for-training`: Feature extraction and model training
- `6-training-process`: Model evaluation and validation
- `7-create-streamlit-dashboard`: Dashboard implementation

### Adding New Features

1. Create a new branch: `git checkout -b feature-name`
2. Implement changes
3. Test thoroughly
4. Commit and push: `git push origin feature-name`
5. Create pull request for review

### Testing

**Unit Tests**:
```bash
python test_upload_system.py
python test_system_comprehensive.py
```

**Manual Testing Checklist**:
- [ ] User authentication (sign up, login, logout)
- [ ] Case creation and analysis
- [ ] Case management (view, update, filter)
- [ ] Intervention tracking
- [ ] Teacher request workflow
- [ ] Analytics dashboard
- [ ] Role-based access control
- [ ] Model predictions accuracy

### Code Style

- Follow PEP 8 Python style guide
- Use type hints where appropriate
- Document functions with docstrings
- Keep functions focused and modular

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## Contact

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the development team
- Refer to project documentation

## Acknowledgments

- Built for St. Elizabeth Primary School, Mukuru
- Uses Supabase for authentication
- Powered by scikit-learn and Streamlit

---

**Last Updated**: [Current Date]
**Version**: 1.0.0
**Status**: Production Ready

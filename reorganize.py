"""
ML Models Repository Reorganization Script
Automatically reorganizes your GitHub repositories professionally
"""

import os
import shutil
import subprocess
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.BLUE}→ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def run_command(cmd, cwd=None):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def create_gitignore():
    """Create comprehensive .gitignore file"""
    return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/
.venv
.env

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Operating System
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Database
*.db
*.sqlite
*.sqlite3

# Testing
.pytest_cache/
.coverage
htmlcov/

# Node modules
node_modules/
package-lock.json

# Temporary files
*.tmp
*.temp
*.bak
"""

def create_requirements_txt():
    """Create requirements.txt file"""
    return """# Core ML & Data Science
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Web Frameworks
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
streamlit>=1.28.0

# Database
psycopg2-binary>=2.9.0
sqlalchemy>=2.0.0

# ML Model Persistence
joblib>=1.3.0

# Image Processing
Pillow>=10.0.0

# Data Validation
pydantic>=2.0.0

# HTTP Requests
requests>=2.31.0

# Jupyter Notebooks
jupyter>=1.0.0
ipykernel>=6.25.0

# Development Tools
pytest>=7.4.0
python-dotenv>=1.0.0
"""

def create_main_readme():
    """Create professional main README"""
    return """<div align="center">

# 🤖 Machine Learning Projects Portfolio

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive collection of machine learning projects showcasing end-to-end ML pipelines, from data preprocessing to model deployment.

</div>

---

## 📋 Projects

### ❤️ Heart Disease Prediction
Predict the risk of heart disease using Random Forest Classifier with FastAPI and Streamlit deployment.

**Tech Stack:** scikit-learn, FastAPI, Streamlit, PostgreSQL  
**Accuracy:** ~85%

📁 `projects/heart-disease-prediction/`

### 📈 Linear Regression
Multiple linear regression models for salary and product sales prediction.

**Tech Stack:** scikit-learn, pandas, matplotlib  
**Models:** Salary prediction, Product sales forecasting

📁 `projects/linear-regression/`

### 🌳 Decision Tree & Random Forest
Tree-based classification models using the Titanic dataset.

**Tech Stack:** scikit-learn, pandas, matplotlib  
**Accuracy:** 82-85% (Random Forest)

📁 `projects/decision-tree/`

### 🖼️ Image Compression (SVD)
Compress images using Singular Value Decomposition.

**Tech Stack:** NumPy, Matplotlib, PIL  
**Technique:** Matrix factorization

📁 `projects/image-compression-svd/`

---

## 🛠️ Tech Stack

**ML & Data Science:**  
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**Web Frameworks:**  
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

**Database:**  
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/MKarthik730/ml-models.git
cd ml-models

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

Each project has its own README with detailed instructions. Navigate to the project folder and follow the setup guide.

Example for Heart Disease Prediction:
```bash
cd projects/heart-disease-prediction
python src/train.py
```

---

## 📂 Repository Structure

```
ml-models/
├── projects/
│   ├── heart-disease-prediction/
│   ├── linear-regression/
│   ├── decision-tree/
│   └── image-compression-svd/
├── notebooks/
├── utils/
├── requirements.txt
└── README.md
```

---

## 📞 Contact

**Karthik Motupalli**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/karthik-motupalli)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MKarthik730)

**Organization:** ANITS Vizag  
**Location:** Visakhapatnam, Andhra Pradesh, India

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

![Profile Views](https://komarev.com/ghpvc/?username=mkarthik730&label=Profile%20views&color=0e75b6&style=flat)

</div>
"""

def reorganize_ml_models():
    """Reorganize ml-models repository"""
    print_header("Reorganizing ML-Models Repository")
    
    repo_path = Path("ml-models")
    if not repo_path.exists():
        print_error("ml-models directory not found!")
        print_info("Please run this script from the parent directory containing ml-models/")
        return False
    
    os.chdir(repo_path)
    
    # Check if it's a git repo
    if not Path(".git").exists():
        print_error("Not a git repository!")
        return False
    
    print_info("Creating backup branch...")
    run_command("git checkout -b backup-before-reorganization")
    run_command("git checkout main")
    
    print_info("Creating new directory structure...")
    
    # Create new folders
    folders = [
        "projects/heart-disease-prediction/data",
        "projects/heart-disease-prediction/models",
        "projects/heart-disease-prediction/src",
        "projects/heart-disease-prediction/api",
        "projects/heart-disease-prediction/app",
        "projects/linear-regression/data",
        "projects/linear-regression/src",
        "projects/decision-tree/data",
        "projects/decision-tree/src",
        "projects/image-compression-svd/images",
        "projects/image-compression-svd/src",
        "notebooks",
        "utils"
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
    
    print_success("Directory structure created")
    
    # Move Heart Disease files
    print_info("Moving Heart Disease Prediction files...")
    heart_moves = [
        ("heart/heart_model.joblib", "projects/heart-disease-prediction/models/heart_model.joblib"),
        ("heart/main.py", "projects/heart-disease-prediction/api/main.py"),
        ("heart/database.py", "projects/heart-disease-prediction/api/database.py"),
        ("heart/databasemodels.py", "projects/heart-disease-prediction/api/databasemodels.py"),
        ("heart/heart-streamlit.py", "projects/heart-disease-prediction/app/streamlit_app.py"),
        ("heart/model2.py", "projects/heart-disease-prediction/src/train.py"),
        ("heart/decision_tree.py", "projects/heart-disease-prediction/src/predict.py"),
        ("heart/data.py", "projects/heart-disease-prediction/src/data_processing.py"),
        ("heart/chat.py", "projects/heart-disease-prediction/src/chat.py"),
        ("heart/index.html", "projects/heart-disease-prediction/app/index.html"),
    ]
    
    for src, dst in heart_moves:
        if Path(src).exists():
            run_command(f'git mv "{src}" "{dst}"')
    
    # Move Linear Regression files
    print_info("Moving Linear Regression files...")
    lr_moves = [
        ("linear_regression.py", "projects/linear-regression/src/linear_regression.py"),
        ("salary_data.csv", "projects/linear-regression/data/salary_data.csv"),
        ("capping_flooring_practice_multi.csv", "projects/linear-regression/data/capping_flooring_practice_multi.csv"),
    ]
    
    for src, dst in lr_moves:
        if Path(src).exists():
            run_command(f'git mv "{src}" "{dst}"')
    
    # Move Decision Tree files
    print_info("Moving Decision Tree files...")
    dt_moves = [
        ("models/decision_tree.py", "projects/decision-tree/src/decision_tree.py"),
        ("models/random-forest.py", "projects/decision-tree/src/random_forest.py"),
        ("models/titanic.csv", "projects/decision-tree/data/titanic.csv"),
        ("models/capping_flooring_practice_multi.csv", "projects/linear-regression/data/capping_flooring_practice_multi.csv"),
    ]
    
    for src, dst in dt_moves:
        if Path(src).exists():
            run_command(f'git mv "{src}" "{dst}"')
    
    # Move SVD file
    print_info("Moving SVD Compression files...")
    if Path("svd.py").exists():
        run_command('git mv "svd.py" "projects/image-compression-svd/src/svd_compression.py"')
    
    # Create .gitkeep files
    print_info("Creating .gitkeep files...")
    gitkeep_files = [
        "projects/heart-disease-prediction/data/.gitkeep",
        "projects/image-compression-svd/images/.gitkeep",
        "notebooks/.gitkeep",
    ]
    
    for file in gitkeep_files:
        Path(file).touch()
    
    Path("utils/__init__.py").touch()
    
    # Remove cached files
    print_info("Removing cached files...")
    run_command("git rm -r __pycache__")
    run_command("git rm -r heart/__pycache__")
    run_command("git rm -r models/__pycache__")
    
    # Create new files
    print_info("Creating configuration files...")
    
    with open(".gitignore", "w") as f:
        f.write(create_gitignore())
    
    with open("requirements.txt", "w") as f:
        f.write(create_requirements_txt())
    
    with open("README.md", "w") as f:
        f.write(create_main_readme())
    
    # Stage all changes
    print_info("Staging all changes...")
    run_command("git add .")
    
    print_success("ML-Models repository reorganized!")
    
    os.chdir("..")
    return True

def main():
    print_header("🚀 Repository Reorganization Tool")
    print_info("This script will reorganize your repositories professionally")
    print_warning("Make sure you're in the parent directory containing your repos")
    
    input("\nPress Enter to continue...")
    
    # Reorganize ml-models
    if reorganize_ml_models():
        print_success("\n✅ Reorganization complete!")
        print_info("\nNext steps:")
        print("1. Review changes: cd ml-models && git status")
        print("2. Commit: git commit -m 'Reorganize repository structure professionally'")
        print("3. Push: git push origin main")
        print("\n" + "="*60)
    else:
        print_error("\n❌ Reorganization failed!")
        print_info("Please check the errors above and try again")

if __name__ == "__main__":
    main()
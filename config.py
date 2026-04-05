import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Flask settings
class Config:
    SECRET_KEY = 'churn-prediction-secret-key-2024'
    DEBUG = True

    # Database
    DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'churn.db')

    # Folders
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'uploads')
    MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
    CHARTS_FOLDER = os.path.join(BASE_DIR, 'static', 'charts')

    # Allowed file types for CSV upload
    ALLOWED_EXTENSIONS = {'csv'}

    # Model settings
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Risk score thresholds
    LOW_RISK = 30       # below 30% = Low risk
    MEDIUM_RISK = 60    # 30–60% = Medium risk
                        # above 60% = High risk
@echo off
echo ========================
echo NLP PROJECT PIPELINE RUN
echo =========================

REM --- Activate Virtual Environment ---
call venv\Scripts\activate

echo.
echo --- Step 1: Running preprocessing ---
python src\preprocess.py

echo.
echo --- step 2: Generating TF-IDF Features ---
python src\features_tfidf.py

echo.
echo --- step 3: Training Model ---
python src\train_model_tfidf.py

echo.
echo --- step 4: Running Prediction on new_data.csv ---
python src\predict_tfidf.py

echo.
echo =============================
echo   ALL TASKS COMPLETED
echo ==============================
pause
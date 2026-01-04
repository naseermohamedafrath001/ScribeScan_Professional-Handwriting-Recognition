# ScribeScan | Professional Handwriting Recognition
ScribeScan is a comprehensive web-based application designed to identify student IDs from handwriting samples using Deep Learning. This project provides a bridge between traditional handwritten work and digital record-keeping, allowing educators to instantly identify authors of submitted assignments. It serves as a dynamic platform for managing student identification, processing bulk handwriting data, and fostering an efficient grading workflow.

![ScribeScan Dashboard](file:///c:/Users/imtha/Documents/My/DL/Deep%20learning%20Project%20n/screenshot.png)

## Features
- **Neural Identity Detection:** Uses a custom Deep CNN to match handwriting patterns to student IDs.
- **Majority Voting Logic:** Aggregates predictions from multiple segments of a single paper for high-precision results.
- **Real-Time Web Analysis:** Asynchronous processing for instant feedback in the browser.
- **Micro-Segmentation:** Automated extraction of characters and words from scanned documents.

## User Types
- **Admin:** Full control over model training, batch processing, and system optimization.
- **User/Educator:** Can upload student work, analyze handwriting, and view identification results.

![Interface Preview](file:///c:/Users/imtha/Documents/My/DL/Deep%20learning%20Project%20n/screenshot.png)

## User Side
- **Handwriting Upload:** Simple drag-and-drop interface for document scanning.
- **Real-Time Analysis:** Click-to-analyze feature using the global neural network.
- **Identification Badge:** Instant display of the identified Student ID with a confidence percentage.
- **Analysis Metrics:** View how many segments the AI used to reach its conclusion.

## Admin Side
- **System Dashboard:** Overview of the model's performance and current class mappings.
- **Batch Evaluation:** Process entire directories of student work and export results to CSV.
- **Model Training:** Retrain the Deep CNN on new handwriting samples to expand system coverage.
- **Data Augmentation:** Configure spatial jitter and noise to improve model robustness.

![ScribeScan Analysis](file:///c:/Users/imtha/Documents/My/DL/Deep%20learning%20Project%20n/screenshot.png)

## Demo
A short demonstration of the ScribeScan system:


## Technology Stack
- **Backend:** Python (Flask)
- **Deep Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV
- **Frontend:** HTML5, Premium Vanilla CSS, JavaScript (ES6)

## Installation Guide
1. **Prerequisites:**
   - Install Python 3.10+ and pip.
   - Install required libraries: `pip install tensorflow opencv-python flask tqdm`
2. **Download:**
   - Clone or download the project source code.
3. **Model Setup:**
   - Ensure `model.h5` and `labels.json` are in the root directory.
4. **Running the Application:**
   - Open your terminal in the project folder.
   - Run the command: `python app.py`
   - Access the site at: `http://127.0.0.1:5000`

## Configuration
The system configuration is handled via command-line arguments in the standalone scripts:
- **Training:** `python train.py --epochs 80 --batch_size 64`
- **Inference:** `python run.py --test_dir "test_dir"`

## Contributors
Anshath Ahamed Ajumil, Mohamed Nawran, Sharaf Sahir, Mohamed Afrath 

⚠️ **Note:** The current model is calibrated for identifying specific student IDs provided during training. For new students, the system must be retrained using `train.py`. 

**License:** Distributed under the MIT License.

**Contact:**
Anshath Ahamed Ajumil - anshath7@gmail.com
Mohamed Nawran - mhdnawran4@gmail.com
Sharaf Sahir - sharafsakeer3333@gmail.com
Mohamed Afrath - mohamednaseermohamedafrath@gmail.com

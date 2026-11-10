PAUS HEALTHCARE CLAIMS INTELLIGENCE APP

CONTENTS
- app.py: Streamlit web application
- portable_scoring.py: portable individual-claim Gradient Boosting scorer
- models/fraud_model_portable.json: model parameters and decision trees (no pickle loading)
- requirements.txt: runtime dependencies
- PROJECT_SUMMARY.md: project methods, results and presentation-ready summary

RUN ON WINDOWS
1. Extract the ZIP file to a folder.
2. Open Command Prompt in the extracted folder (the folder containing app.py).
3. Run:  py -m pip install -r requirements.txt
4. Run:  py -m streamlit run app.py
5. The app opens in your browser (usually http://localhost:8501).

APP FUNCTIONS
- Claim amount estimate: exact six-effect scoring equation supplied from SAS GLMSELECT.
- Fraud risk screening: individual claim form scored by the bundled portable Gradient Boosting model.
- Model performance: shows SAS Viya model comparison results separately from local app model metrics.
- Methodology & notes: documents model provenance and limitations.

MODEL PROVENANCE
The individual fraud form uses a locally packaged Gradient Boosting model fitted on the supplied 10,000-row dataset, using the 14 classification inputs and a stratified 70:30 split. The scorer is exported into portable JSON decision-tree parameters and does not load a Python pickle, avoiding scikit-learn/joblib version compatibility failures. It is an independently fitted app model aligned to the exported SAS training settings, not the registered SAS Viya model binary. SAS Viya metrics are displayed separately and must not be attributed to the local app model. The dataset is simulated. Use outputs only for decision support.

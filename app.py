import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title='Healthcare Claims Intelligence', page_icon='🏥', layout='wide')
from portable_scoring import predict_fraud_probability, FEATURES as fraud_features, MODEL as portable_model
local_metrics = portable_model['metrics']

# Exact scoring coefficients supplied by the SAS GLMSELECT scoring logic.
B = {'intercept':522.602238691581,'female':20.476514735386,'private':-12.9555790325701,
     'internal_medicine':13.5467511953848,'inpatient':-4.01611408761764,
     'days_between':-3.55207938214922,'claims_per_month':0.39264475395121}

st.title('Healthcare Claims Intelligence')
st.caption('PAUS Project | Claim Amount Estimation & Fraud Risk Screening')
st.info('Dataset: simulated healthcare claims data. Outputs are analytical estimates for decision support, not claim approvals or definitive fraud determinations.')

reg_tab, fraud_tab, dashboard_tab, methodology_tab = st.tabs(['Claim amount estimate','Fraud risk screening','Model performance','Methodology & notes'])

with reg_tab:
    st.subheader('Estimate the submitted claim amount')
    st.write('Enter the six variables used by the retained SAS regression scoring equation.')
    with st.form('reg_form'):
        a,b=st.columns(2)
        with a:
            gender=st.selectbox('Patient gender',['Female','Male'])
            insurance=st.selectbox('Insurance type',['Medicaid','Medicare','Private','Self-Pay'])
            specialty=st.selectbox('Provider specialty',['Cardiology','General Practice','Internal Medicine','Neurology','Orthopedics','Pulmonology'])
        with b:
            visit=st.selectbox('Visit type',['Emergency','Inpatient','Outpatient'])
            days=st.number_input('Days between service and claim',min_value=0,max_value=3650,value=7,step=1)
            claims=st.number_input('Claims per provider per month',min_value=0,max_value=100000,value=10,step=1)
        reg_submit=st.form_submit_button('Calculate estimate',type='primary')
    if reg_submit:
        terms={'Intercept':B['intercept'],'Female':B['female'] if gender=='Female' else 0,
               'Private insurance':B['private'] if insurance=='Private' else 0,
               'Internal Medicine':B['internal_medicine'] if specialty=='Internal Medicine' else 0,
               'Inpatient':B['inpatient'] if visit=='Inpatient' else 0,
               'Days between service and claim':B['days_between']*days,
               'Claims per provider per month':B['claims_per_month']*claims}
        amount=sum(terms.values())
        st.metric('Estimated claim amount',f'₹ {amount:,.2f}')
        st.caption('Calculated using the SAS-generated regression scoring coefficients provided for this project.')
        d=pd.DataFrame({'Model term':terms.keys(),'Contribution (₹)':terms.values()})
        fig,ax=plt.subplots(figsize=(8,3.5)); ax.barh(d['Model term'],d['Contribution (₹)']); ax.axvline(0,linewidth=.8); ax.set_xlabel('Contribution (₹)'); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        with st.expander('Calculation details'):
            st.dataframe(d,use_container_width=True,hide_index=True)

with fraud_tab:
    st.subheader('Screen a claim for potential fraud')
    st.write('Enter the claim and provider characteristics below. The app returns a fraud-risk probability and a model classification.')
    with st.form('fraud_form'):
        c1,c2,c3=st.columns(3)
        with c1:
            claim_amount=st.number_input('Claim amount (₹)',min_value=0.0,value=1000.0,step=50.0)
            days_service=st.number_input('Days between service and claim',min_value=0,value=7,step=1)
            claims_month=st.number_input('Claims per provider per month',min_value=0,value=10,step=1)
            age=st.number_input('Patient age',min_value=0,max_value=120,value=45,step=1)
            chronic=st.selectbox('Chronic condition flag',[0,1],format_func=lambda x:'Yes' if x==1 else 'No')
        with c2:
            gender_f=st.selectbox('Patient gender',['Female','Male'])
            diagnosis=st.selectbox('Diagnosis code',['E11.9','E78.5','F41.9','I10','I25.10','J06.9','J18.9','K21.9','M54.5','N39.0'])
            insurance_f=st.selectbox('Insurance type',['Private','Medicare','Medicaid','Self-Pay'])
            los=st.number_input('Length of stay (days)',min_value=0,max_value=365,value=1,step=1)
            state=st.selectbox('Patient state',['CA','FL','GA','IL','NY','OH','PA','TX'])
        with c3:
            prior=st.number_input('Prior visits in last 12 months',min_value=0.0,value=2.0,step=1.0)
            procedure=st.number_input('Procedure code',min_value=0,value=100,step=1)
            specialty_f=st.selectbox('Provider specialty',['Cardiology','General Practice','Internal Medicine','Neurology','Orthopedics','Pulmonology'])
            visit_f=st.selectbox('Visit type',['Emergency','Inpatient','Outpatient'])
        fraud_submit=st.form_submit_button('Assess fraud risk',type='primary')
    if fraud_submit:
        row=pd.DataFrame([{'Claim_Amount':claim_amount,'Days_Between_Service':days_service,
            'Number_of_Claims_Per_Provider_Mo':claims_month,'Patient_Age':age,'Chronic_Condition_Flag':chronic,
            'Patient_Gender':gender_f,'Diagnosis_Code':diagnosis,'Insurance_Type':insurance_f,
            'Length_of_Stay':los,'Patient_State':state,'Prior_Visits_12m':prior,'Procedure_Code':procedure,
            'Provider_Specialty':specialty_f,'Visit_Type':visit_f}],columns=fraud_features)
        p=float(predict_fraud_probability(row.iloc[0].to_dict())); pred=int(p>=.5)
        st.metric('Estimated fraud probability',f'{p:.1%}')
        if pred: st.error('Model classification: Potentially fraudulent — manual review recommended.')
        else: st.success('Model classification: Likely genuine — standard review process still applies.')
        st.progress(p)
        st.caption('This is a locally packaged Gradient Boosting model trained on the supplied 10,000-row dataset. It is not a live call to the registered SAS Viya model.')

with dashboard_tab:
    st.subheader('Classification results from SAS Viya Model Comparison')
    sas=pd.DataFrame([
      ['Gradient Boosting',94.89,.9632,.5521,.8388],['Pruned Decision Tree',95.24,.9469,.5310,.8260],
      ['Full Decision Tree',94.71,.9545,.5546,.8172],['Random Forest',95.17,.9649,.5141,.8233],
      ['Logistic Regression',92.99,.5000,0,.0000]],columns=['Model','Accuracy (%)','AUC-ROC','F1 score','KS'])
    st.dataframe(sas,use_container_width=True,hide_index=True)
    st.caption('SAS Viya selected Gradient Boosting as champion by highest KS statistic (0.8388). Random Forest had the highest AUC-ROC (0.9649); Full Decision Tree had the highest F1 (0.5546); Pruned Decision Tree had the highest accuracy (95.24%).')
    st.divider(); st.subheader('Packaged app model — independent local reproduction')
    m=local_metrics
    k1,k2,k3,k4=st.columns(4)
    k1.metric('Accuracy',f"{m['accuracy']:.2%}"); k2.metric('AUC-ROC',f"{m['auc_roc']:.4f}"); k3.metric('F1 score',f"{m['f1']:.4f}"); k4.metric('Recall',f"{m['recall']:.2%}")
    st.caption(f"Local model test set: {m['test_rows']:,} rows; {m['split']}; threshold {m['threshold']:.2f}. These are not the SAS Viya report metrics and should not be presented as such.")
    st.warning('Fraud prevalence is low (8.29% in the full dataset). Accuracy alone can be misleading; review precision, recall, F1 and AUC-ROC.')

with methodology_tab:
    st.subheader('Project scope and implementation')
    st.markdown('''
**Business objective:** estimate submitted healthcare claim amounts and flag claims that may require additional fraud review.

**Dataset:** 10,000 simulated healthcare claim records. The data are synthetic and do not represent actual patients or insurance claims.

**Regression:** uses the supplied SAS GLMSELECT scoring equation with six predictors/effects. `Approved_Amount`, identifiers and post-outcome fields are not used in the displayed equation.

**Classification:** uses the 14 predictors used in the SAS Viya Gradient Boosting workflow. A locally packaged Gradient Boosting pipeline is included so the form can return an individual prediction without exposing SAS credentials or infrastructure settings.

**Important distinction:** the packaged app classifier is a local re-fit on the supplied dataset, using a stratified 70:30 split and GradientBoostingClassifier settings aligned to the exported SAS training settings (100 estimators, learning rate 0.1, subsample 0.5, max depth 4, minimum leaf size 5). It is not the exact registered SAS Viya model binary. SAS Viya model comparison metrics are shown separately.

**Operational use:** treat the output as a screening signal. A human reviewer must assess supporting documentation before any claim decision.
''')
    st.subheader('Classification input fields')
    st.write(', '.join(fraud_features))
    st.subheader('Regression model equation')
    st.code('''Predicted Claim_Amount = 522.6022
 + 20.4765 × Female
 - 12.9556 × Private insurance
 + 13.5468 × Internal Medicine
 - 4.0161 × Inpatient
 - 3.5521 × Days between service and claim
 + 0.3926 × Claims per provider per month''')

st.divider(); st.caption('PAUS Project | Healthcare Claims Intelligence | Simulated data | Decision-support only')

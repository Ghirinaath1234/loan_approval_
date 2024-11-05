import streamlit as st
from doctr.io import DocumentFile
import pdfplumber
import pickle
from imageout import paper_output,normalize_ocr_data
from PIL import Image
import numpy as np
import pandas as pd

st.title("Loan Approval System")

details_dict={}
loan_id=[]
model_list=[]
applicant=[]


with open('bank.model','rb') as my_model:
    data=pickle.load(my_model)
model=data['model']


uploaded_file=st.file_uploader("Choose an document or image...", type=["jpg", "png", "jpeg", "pdf"],accept_multiple_files=True)
if uploaded_file:
    for uploaded_file in uploaded_file:
            if uploaded_file.name.endswith(".pdf"):
                with pdfplumber.open(uploaded_file) as pdf:
                            for page in pdf.pages:
                                text = page.extract_text()
                input_pdf=[]
                for i in text.split():
                    if i.isdigit():
                        input_pdf.append(int(i))
                model_output=model.predict([input_pdf])
                applicant.append(text.split()[5])
                loan_id.append(text.split()[8])
                for i in model_output:
                    if i==' Approved' in model_output:
                        model_list.append('Approval')
                    elif i==' Rejected' in model_output:
                        model_list.append('Not Approval')  
            else:
                doc=DocumentFile.from_images(uploaded_file.name)
                json_output=paper_output(doc)
                normal_input=normalize_ocr_data(json_output)
                input_arr=[]
                for i in normal_input:
                    if i.isdigit():
                        input_arr.append(int(i))
                model_output=model.predict([input_arr])
                applicant.append(normal_input[2])
                loan_id.append(normal_input[4])
                for i in model_output:
                    if i==' Approved' in model_output:
                         model_list.append('Approval')
                    elif i==' Rejected' in model_output:
                         model_list.append('Not Approval')
                         
details_dict['LoanID']=loan_id
details_dict['Name']=applicant
details_dict['loan_status']=model_list

result=st.button("Submit")

if result:
    df=pd.DataFrame(details_dict)
    df=df.drop_duplicates()
    st.dataframe(df)
    csv_full=df.to_csv(index=False).encode('utf-8')
    df_approval=df[df['loan_status']=='Approval']
    csv_approval=df_approval.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download full applicant sheet",data=csv_full,file_name='loan_applicant.csv',mime='text/csv')
    st.download_button(label="Download Approved applicant sheet",data=csv_approval,file_name='approved_applicant.csv',mime='text/csv')
    
    

    

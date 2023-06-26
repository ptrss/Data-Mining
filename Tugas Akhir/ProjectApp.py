import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth


# Method(s)
##=======================================================================
def hot_encode(x):
    if(x <= 0):
        return 0
    if(x > 0):
        return 1
    
# Button Session
##======================================================================
if "prepareBtn" not in st.session_state:
    st.session_state["prepareBtn"] = False
if "encodeBtn" not in st.session_state:
    st.session_state["encodeBtn"] = False
if "associationSlider" not in st.session_state:
    st.session_state["associationSlider"] = False
    
#=======================================================================
st.title("Association w/ Apriori")

## Import Dataset
#=======================================================================
st.header('Importing Dataset')

data = pd.DataFrame()
uploaded_file = st.file_uploader('Upload Dataset', type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Read the file
    file_type = uploaded_file.name.split('.')[-1]
    if file_type=='csv':
        data = pd.read_csv(uploaded_file, encoding='utf-8')
    elif file_type=='xlsx':
        data = pd.read_excel(uploaded_file)
        
# Button Prepare
##======================================================================    
if st.button("Prepare"):
    st.session_state["prepareBtn"] = not st.session_state["prepareBtn"]
    
if st.session_state["prepareBtn"]:
    st.header('Data Prepared')
    st.write("Applied:\n1. Seleksi kolom\n2. Menghapus baris yang tidak memiliki Invoice/Nota\n3. Mengubah data type kolom 'Invoice'\n4. Menghilangkan extra white space\n5. Encode data")
    
    data = data[['Invoice', 'NamaPerusahaan', 'NamaBarang', 'Qty']]
    data.dropna(axis=0, subset=['Invoice'], inplace=True)
    data['Invoice'] = data['Invoice'].astype('str').str.split('.').str[0]
    data['NamaPerusahaan'] = data['NamaPerusahaan'].str.strip()
    data['NamaBarang'] = data['NamaBarang'].str.strip()
    
    data = (data.groupby(['Invoice', 'NamaBarang'])['Qty']
                .sum().unstack().reset_index().fillna(0)
                .set_index('Invoice'))
    data = data.applymap(hot_encode)
    
# Association Rules
##======================================================================
if st.session_state["prepareBtn"]:
    if st.button("Create Association Rules"):
        st.session_state["associationSlider"] = not st.session_state["associationSlider"]
        
if st.session_state["associationSlider"]:
    st.header('Association Rule Mining')
    st.write('Discovering associations between items')

    ## Sidebar set minimum support and confidence thresholds
    min_support = st.slider('Minimum Support', min_value=0.002, max_value=0.005, value=0.002, step=0.001, format="%.3f")
    min_confidence = st.slider('Minimum Confidence', min_value=0.1, max_value=1.0, value=0.3, step=0.1)

    ## Generate frequent itemsets using Apriori algorithm
    frq_items = apriori(data, min_support=min_support, use_colnames=True)

    ## Generate association rules
    rules = association_rules(frq_items, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(['confidence'], ascending=[False])

    ## Display the generated rules
    st.subheader('Apriori Bottom 5 Rules')
    st.dataframe(rules.head(5))
    
    st.subheader('Apriori Top 5 Rules')
    st.dataframe(rules.tail(5))
    
# Input Nama Barang and Display Consequent Items
##======================================================================
if st.session_state["associationSlider"]:
    st.header('Input Nama Barang')
    nama_barang = st.text_input("Masukkan nama barang:")

    if st.button("Cari Barang Consequent"):
        consequent_barang = rules[rules['antecedents'].apply(lambda x: any(nama_barang in item for item in x))]
        consequent_barang['Consequent Item'] = consequent_barang['consequents'].apply(lambda x: [item for item in x if nama_barang not in item])
                                              
        filtered_rules = consequent_barang[consequent_barang['Consequent Item'].apply(lambda x: len(x) > 0)]

        if consequent_barang.empty:
            st.write("Tidak ada barang consequent yang ditemukan.")
        else:
            st.write("Barang consequent yang ditemukan:")
            consequent_list = []
            for item in filtered_rules['Consequent Item']:
                consequent_list.extend(item)
            consequent_barang = pd.DataFrame({'Daftar Barang Consequent' : list(set(consequent_list))})
            st.dataframe(filtered_rules[['Consequent Item', 'support', 'confidence', 'lift']])
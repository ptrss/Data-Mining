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
st.header('Importing Dataset')
data = pd.read_csv('D:\\Me\\Ptrs\\Kuliah\\Smt4\\DataMining\\Project\\Penjualan2017.csv')
st.dataframe(data)
st.write(f"Data shape {data.shape}")
st.write("#### Checking null values")
st.dataframe(data.isnull().sum().reset_index().rename(columns={'index': 'Column', 0: 'Null Count'}))

# Button Prepare
#=======================================================================    
if st.button("Prepare"):
    st.session_state["prepareBtn"] = not st.session_state["prepareBtn"]
    
if st.session_state["prepareBtn"]:
    st.header('Data Prepared')
    st.write("Applied:\n1. Seleksi kolom\n2. Menghapus baris yang tidak memiliki Invoice/Nota\n3. Mengubah data type kolom 'Invoice'\n4. Menghilangkan extra white space")
    data = data[['Invoice', 'NamaPerusahaan', 'NamaBarang', 'Qty']]
    data.dropna(axis=0, subset=['Invoice'], inplace=True)
    data['Invoice'] = data['Invoice'].astype('str').str.split('.').str[0]
    data['NamaPerusahaan'] = data['NamaPerusahaan'].str.strip()
    data['NamaBarang'] = data['NamaBarang'].str.strip()
    st.dataframe(data)

# Button Encode
#=======================================================================
if st.session_state["prepareBtn"]:
    if st.button("Encode"):
        st.session_state["encodeBtn"] = not st.session_state["encodeBtn"]

if st.session_state["encodeBtn"]:
    st.header('Encoded')
    st.write("Applied:\n1. Menampilkan jumlah barang terjual dalam satu Invoice/Nota\n2. Hot Encoder to mark sold items(1) and unsold(0)")
    data = (data.groupby(['Invoice', 'NamaBarang'])['Qty']
                    .sum().unstack().reset_index().fillna(0)
                    .set_index('Invoice'))
    data = data.applymap(hot_encode)
    st.dataframe(data)
    
# Association Rules
#=======================================================================
if st.session_state["prepareBtn"] and st.session_state["encodeBtn"]:
    if st.button("Create Rules"):
        st.session_state["associationSlider"] = not st.session_state["associationSlider"]
        
if st.session_state["associationSlider"]:
    st.header('Association Rule Mining')
    st.write('Discovering associations between items')

    ## Sidebar set minimum support and confidence thresholds
    min_support = st.slider('Minimum Support', min_value=0.002, max_value=0.005, value=0.002, step=0.001, format="%.3f")
    min_confidence = st.slider('Minimum Confidence', min_value=0.1, max_value=1.0, value=0.3, step=0.1)

    ## Generate frequent itemsets using Apriori algorithmx
    frq_items = apriori(data, min_support=min_support, use_colnames=True)

    ## Generate association rules
    rules = association_rules(frq_items, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(['confidence'], ascending=[False])

    ## Display the generated rules
    st.subheader('Apriori Bottom 5 Rules')
    st.dataframe(rules.tail(5))
    
    st.subheader('Apriori Top 5 Rules')
    st.dataframe(rules.head(5))
    
# Input Nama Barang and Display Consequent Items
#=======================================================================
if st.session_state["associationSlider"]:
    st.header('Imput Nama Barang')
    nama_barang = st.text_input("Masukkan nama barang:")

    if st.button("Cari Barang Consequent"):
        consequent_barang = rules[rules['consequents'].apply(lambda x: any(nama_barang in item for item in x))]
        consequent_barang = consequent_barang['consequents'].apply(lambda x: [item for item in x if nama_barang not in item])

        if consequent_barang.empty:
            st.write("Tidak ada barang consequent yang ditemukan.")
        else:
            st.write("Barang consequent yang ditemukan:")
            for item in consequent_barang:
                st.write(item)


# st.write(
#     f"""
#     ## Session state:
#     {st.session_state["prepareBtn"]=}

#     {st.session_state["encodeBtn"]=}

#     {st.session_state["associationSlider"]=}
#     """
# )
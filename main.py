import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
# st.set_page_config(layout='wide')

# Potential Fraud Records versus None Fraudulent Records
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('Data/fraud_none_fraud.csv')
    return df 

st.subheader("Potential Fraud Records VS Non Fraudulent Records")
st.markdown(
    "Both the bar chart and the pie chart help us understand the magnitude of fraudulent and non-fraudulent records.\
    As a result, there are fewer records that are relevant to healthcare fraud. \
    The goal of the project aims to find interesting features or patterns that \
    could help identify fraudulent records from non-fraudulent records."
    )

fraud_none_fraud = load_data()
fig,axes = plt.subplots(1,2,figsize=(8,5))
ttl_data= [fraud_none_fraud['PotentialFraud'].value_counts().Yes, fraud_none_fraud['PotentialFraud'].value_counts().No]

labels = ['Potential Fraud','Non Fraud']; palette = ['#ed544a','#f5d576']; explode = (0, 0.1) 

axes[0].pie(ttl_data, labels=labels,colors = palette,autopct = '%0.0f%%',shadow=True,explode=explode)
sns.barplot(x=labels,y=ttl_data,palette=palette,ax=axes[1])

axes[1].set_title("Actual Count of Labels")
axes[1].bar_label(container=axes[1].containers[0], labels=list(ttl_data))

st.pyplot(fig)
st.markdown("---")
###############################################################################################################################
from mlxtend.frequent_patterns import apriori

st.subheader("Based on given labels, do chronic conditions provide insights into distinguishing fraud and non-fraud records?")
@st.cache(allow_output_mutation=True)
def load_full_data():
    df = pd.read_csv('Data/full_df.csv')
    return df

@st.cache(allow_output_mutation=True)
def create_condition_matrix(outpatient_df):
    chronic_condition = [col for col in outpatient_df.columns 
                         if col.startswith('ChronicCond') or col == 'PotentialFraud']
    
    chronic_condition_matrix = outpatient_df[chronic_condition].set_index('PotentialFraud')
    chronic_condition_matrix = chronic_condition_matrix.subtract(1)
    
    return chronic_condition_matrix,chronic_condition

@st.cache(allow_output_mutation=True)
def top_n_itemsets(df,support,top_n):
    matrix,chronic_cols = create_condition_matrix(df)
    chronic_itemsets = apriori(matrix,min_support=support,use_colnames=True)
    chronic_itemsets = chronic_itemsets[chronic_itemsets.itemsets.apply(lambda x:len(x)) > 1]
    top_n_itemsets = chronic_itemsets.sort_values(by='support',ascending=False).nlargest(top_n,columns='support')
    top_n_itemsets = top_n_itemsets.reset_index(drop=True)
    
    return top_n_itemsets

full_df = load_full_data()
pos_top10_condition= top_n_itemsets(full_df[full_df.PotentialFraud == 'Yes'],0.5,10)
neg_top10_condition = top_n_itemsets(full_df[full_df.PotentialFraud == 'No'],0.5,10)

merge_results = pos_top10_condition.merge(neg_top10_condition,how='left',on='itemsets')
merge_results.columns = ['None Fraudulent','itemsets','Fraudulent']
merge_results = merge_results[['itemsets','Fraudulent','None Fraudulent']]
merge_results.columns = ['itemsets','Fraudulent %','None Fraudulent %']
merge_results.itemsets = merge_results.itemsets.apply(set)
st.table(merge_results)
st.markdown("---")
###############################################################################################################################
st.subheader("Can we identify Physicians who regularly engaged in Fraudulent activities?")
st.markdown("**Sanity Check:**\
            \nPhysicians who frequently conduct in fraud are expected to have 0 or less records in None-Fraudulent Records")

@st.cache(allow_output_mutation=True)
def load_phy_in_fraud():
    df = pd.read_csv('Data/physicians_found_in_fraud_records.csv')
    return df

@st.cache(allow_output_mutation=True)
def load_phy_nin_fraud():
    df = pd.read_csv('Data/physicians_not_found_in_fraud_records.csv')
    return df 

phy_in_fraud = load_phy_in_fraud()
phy_nin_fraud = load_phy_nin_fraud()

st.markdown('**The total appearance of each physician who engages in potential fraud activity**')
st.dataframe(phy_in_fraud.head(10))

st.markdown('**The total appearance of each physician who does not engage in  fraud activity**')
st.dataframe(phy_nin_fraud.head(10))

def tier_categorizer(value):
    if value >= eng_mean + 2 * eng_std:
        return 'Tier-1'
    elif (value < eng_mean + 2 * eng_std) & (value > eng_mean + 1 * eng_std):
        return 'Tier-2'
    else:
        return 'Tier-3'
    
eng_mean   = np.mean(phy_in_fraud['Cnt_Fraud_Engagement_Rate'])
eng_std    = np.std(phy_in_fraud['Cnt_Fraud_Engagement_Rate'])
phy_in_fraud['Tier'] = phy_in_fraud['Cnt_Fraud_Engagement_Rate'].apply(tier_categorizer)

st.markdown("---")
st.subheader('**Can we further identify Physicians who regularly engaged in Fraudulent activities in terms of groups?**')
st.markdown("A knowledge graph is composed of nodes and edges, where nodes represent physicians, and edges represent\
relationships. In addition, the color of nodes and edges are based on the total count of occurrences found in \
fraudulent records. Detailed calculation can be found at cell [15] (tier_categorizer) in Inpatient Analysis and cell [17] \
(tier_categorizer) in Outpatient Analysis Notebooks")
st.markdown("As a result, with minimum support of 0.025 using the apriori itemset mining algorithm on fraudulent records,\
            there are more than 30 groups of physicians identified from the outpatient dataset. ")




knowledge_graph = open('Data/outpatient_knowledge_graph_phy_fraud_subgroups.html','r',encoding='utf-8')
# knowledge_graph = load_html()
st.components.v1.html(knowledge_graph.read(),width=800,height=800,scrolling=True)
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly_express as px
import seaborn as sns
import pickle
import shap
import plotly.graph_objects as go
from matplotlib.image import imread
from zipfile import ZipFile

from sklearn.neighbors import NearestNeighbors

# st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

###########################################################################################################################
###########################################################################################################################
header = st.container()
customer = st.container()
score = st.container()
descriptive = st.container()
intrepretation = st.container()
comparaison = st.container()

#Load  data
@st.cache_data
def load_data():
    z = ZipFile("data/X_dashboard.zip")
    X_dashboard = pd.read_csv(z.open("X_dashboard.csv"), index_col='SK_ID_CURR', encoding ='utf-8')  #Currently on my local machine

    z1 = ZipFile("data/dash_df.zip")
    dash_df = pd.read_csv(z1.open("dash_df.csv"), index_col='SK_ID_CURR', encoding ='utf-8')#Currently on my local machine

    z2 = ZipFile("data/HomeCredit_columns_description.zip")
    description_df = pd.read_csv(z2.open('HomeCredit_columns_description.csv'),encoding = "ISO-8859-1", engine='python')
        
    logo = imread("data/pret_logo.png")


    return X_dashboard, dash_df, description_df, logo

X_dashboard, dash_df, description_df, logo = load_data()

###########################################################################################################################
###########################################################################################################################
st.sidebar.image(logo)
selection = st.sidebar.radio("Please select your scoring model", ['AUC_Score', 'Bank_Score', 'Fbeta_Score'])
if selection=="AUC_Score":
    th = 0.45
    st.sidebar.markdown('***')
    original_title = '<p style="font-size: 20px; color:Blue; text-align: left "> AUC_Score Selected:\nThreshold of payment dificulty= {} </p>'.format(th)
    st.sidebar.markdown(original_title, unsafe_allow_html=True)
    model = pickle.load(open('LGBMClassifier_auc_score.pkl', 'rb'))
    

elif selection=="Bank_Score":
    th = 0.525
    st.sidebar.markdown('***')
    original_title = '<p style="font-size: 20px; color:Blue; text-align: left "> Bank_Score Selected:\nThreshold of payment dificulty= {} </p>'.format(th)
    st.sidebar.markdown(original_title, unsafe_allow_html=True)
    model = pickle.load(open('LGBMClassifier_bank_score.pkl', 'rb'))
    

elif selection=="Fbeta_Score":
    th = 0.375
    st.sidebar.markdown('***')
    original_title = '<p style="font-size: 20px; color:Blue; text-align: left "> F-beta_Score Selected:\nThreshold of payment dificulty= {} </p>'.format(th)
    st.sidebar.markdown(original_title, unsafe_allow_html=True)
    model = pickle.load(open('LGBMClassifier_fbeta_score.pkl', 'rb'))
    

st.sidebar.markdown('***')

###########################################################################################################################
###########################################################################################################################
with header:
    #st.title('Home Credit Risk Estimator!',)
    original_title = '<p style="font-size: 50px; color:Red; text-align: left "> <u>Home Credit Risk Estimator!</u> </p>'
    st.markdown(original_title, unsafe_allow_html=True)
    st.subheader('Welcome to the Home Credit Risk Estimator Dashboard:')
    st.subheader('Descriptive, Score, Explanation & Comparaison')

###########################################################################################################################
###########################################################################################################################
with customer:

    id_list = X_dashboard.index.values

    st.markdown('***')
    original_title = '<p style="font-size: 25px; color:Blue; text-align: left "> Please select the customer ID : </p>'
    st.markdown(original_title, unsafe_allow_html=True)
 
    id = st.selectbox('customer ID', options=id_list)
    id = int(id)
###########################################################################################################################
###########################################################################################################################
with score:
    
    X = X_dashboard[X_dashboard.index == id]
    probability_default_payment = model.predict_proba(X)[:, 1]
    if probability_default_payment >= th:
        prediction = "Credit Not Accorded"
    else:
        prediction = "Credit Accorded"

    st.markdown('***')    
    original_title = '<p style="font-size: 25px;text-align: left; color:Blue"> Probability of payment dificulties : </p>'
    st.markdown(original_title, unsafe_allow_html=True)
    #st.markdown('***') 

    original_title = '<p style="font-family:Courier; color:BROWN; font-size:50px; text-align: center;">{}%</p>'.format((probability_default_payment[0]*100).round(2))
    st.markdown(original_title, unsafe_allow_html=True)

    # st.markdown('***')    
    original_title = '<p style="font-size: 25px;text-align: left; color:Blue;"> Conclusion : </p>'
    st.markdown(original_title, unsafe_allow_html=True)
    #st.markdown('***')

    def color(pred):
        '''Définition de la couleur selon la prédiction'''
        if pred=='Credit Accorded':
            col='Green'
        else :
            col='Red'
        return col

    fig = go.Figure(go.Indicator(mode = "gauge+number+delta",
                                value = probability_default_payment[0],
                                number = {'font':{'size':48}},
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Customer's Request Status", 'font': {'size': 28, 'color':color(prediction)}},
                                delta = {'reference': th, 'increasing': {'color': "red"},'decreasing':{'color':'green'}},
                                gauge = {'axis': {'range': [0,1], 'tickcolor': color(prediction)},
                                         'bar': {'color': color(prediction)},
                                         'steps': [{'range': [0,th], 'color': 'lightgreen'},
                                                    {'range': [th,1], 'color': 'lightcoral'}],
                                         'threshold': {'line': {'color': "black", 'width': 5},
                                                       'thickness': 1,
                                                       'value': th}}))
    st.plotly_chart(fig)


    if prediction == "Credit Accorded":
        original_title = '<p style="font-family:Courier; color:GREEN; font-size:65px; text-align: center;">{}</p>'.format(prediction)
        st.markdown(original_title, unsafe_allow_html=True)
    else :
        original_title = '<p style="font-family:Courier; color:red; font-size:65px; text-align: center;">{}</p>'.format(prediction)
        st.markdown(original_title, unsafe_allow_html=True)

###########################################################################################################################
###########################################################################################################################
with descriptive:

    st.markdown('***')
    original_title = '<p style="font-size: 25px; color:Blue; text-align: left "> Customer general informations : </p>'
    st.markdown(original_title, unsafe_allow_html=True)

    chk_general = st.checkbox("Show Customer Personal Informations ?", value=True)

    if chk_general:
        
        c1, c2, c3, c4, c5 = st.columns(5)
        with st.container():
            c1.write('**Gender** :' + str(dash_df[dash_df.index==id].CODE_GENDER.values[0]))
            c2.write('**Family Status** : ' + str(dash_df[dash_df.index==id].NAME_FAMILY_STATUS.values[0]))
            c3.write("**Children** : " + str(dash_df[dash_df.index==id].CNT_CHILDREN.values[0]))
            c4.write('**Age** : ' + str(dash_df[dash_df.index==id].DAYS_BIRTH.values[0].round())+ ' years')
            c5.write('**Education** : ' + str(dash_df[dash_df.index==id].NAME_EDUCATION_TYPE.values[0]))   
            st.markdown('')

        c1, c2, c3, c4, c5 = st.columns(5)
        with st.container(): 
            c1.write('**Accompanying customer** : '+str(dash_df[dash_df.index==id].NAME_TYPE_SUITE.values[0]))
            c2.write('**Owning car** : ' + str(dash_df[dash_df.index==id].FLAG_OWN_CAR.values[0]))
            c3.write('**Owning House** : ' + str(dash_df[dash_df.index==id].FLAG_OWN_REALTY.values[0]))
            c4.write("**Housing type** : " + str(dash_df[dash_df.index==id].NAME_HOUSING_TYPE.values[0]))
            c5.write('**Occupation** : ' + str(dash_df[dash_df.index==id].OCCUPATION_TYPE.values[0]))    
            st.markdown('')

        c1, c2, c3, c4, c5 = st.columns(5)
        with st.container(): 
            c1.write('**Organization** : '+str(dash_df[dash_df.index==id].ORGANIZATION_TYPE.values[0]))
            c2.write('**Employment duration** : ' + str(dash_df[dash_df.index==id].DAYS_EMPLOYED.values[0].round())+ ' years')
            c3.write('**Total income** : ' + str(dash_df[dash_df.index==id].AMT_INCOME_TOTAL.values[0]) + ' USD')
            # c4.write("**Loan Type** : " + str(dash_df[dash_df.index==id].NAME_CONTRACT_TYPE.values[0]))
            # c5.write('**Loan amount** : ' + str(dash_df[dash_df.index==id].AMT_CREDIT.values[0]) + ' USD')
        
        pos_1, pos_2, pos_3 = st.columns(3)
        
        with pos_1:
            # @st.cache_resource()
            def plot():                
                fig, ax = plt.subplots()
                sns.kdeplot(data=dash_df, x='DAYS_BIRTH', label = 'Age', hue='CODE_GENDER') #log_scale=True, 
                plt.axvline(x=dash_df[dash_df.index==id].DAYS_BIRTH.values[0], ymax=0.95, color='firebrick', ls='--')
                return fig            
            st.pyplot(plot())

        with pos_2:
            # @st.cache_resource()
            def plot():                
                fig, ax = plt.subplots()
                sns.kdeplot(data=dash_df, x='AMT_INCOME_TOTAL', log_scale=True, hue='FLAG_OWN_REALTY')
                plt.axvline(x=dash_df[dash_df.index==id].AMT_INCOME_TOTAL.values[0], ymax=0.95, color='firebrick', ls='--')
                return fig
            st.pyplot(plot())

        with pos_3:
            # @st.cache_resource()
            def plot():                
                fig, ax = plt.subplots()
                sns.kdeplot(data=dash_df, x='AMT_INCOME_TOTAL', log_scale=True, hue='NAME_EDUCATION_TYPE')
                plt.axvline(x=dash_df[dash_df.index==id].AMT_INCOME_TOTAL.values[0], ymax=0.95, color='firebrick', ls='--')
                return fig
            st.pyplot(plot())

        with pos_1:
            # @st.cache_resource()
            def plot():                
                fig, ax = plt.subplots()
                sns.kdeplot(data=dash_df, x='AMT_INCOME_TOTAL', log_scale=True, hue='FLAG_OWN_CAR')
                plt.axvline(x=dash_df[dash_df.index==id].AMT_INCOME_TOTAL.values[0], ymax=0.95, color='firebrick', ls='--')
                return fig
            st.pyplot(plot())

        with pos_2:
            # @st.cache_resource()
            def plot():                
                fig, ax = plt.subplots()
                splot = sns.scatterplot(x=dash_df['DAYS_BIRTH'], y=dash_df['DAYS_EMPLOYED'], 
                                        hue=dash_df['NAME_FAMILY_STATUS'])
                # splot.set(xscale="log")
                plt.scatter(x=dash_df[dash_df.index==id].DAYS_BIRTH.values[0], 
                            y=dash_df[dash_df.index==id].DAYS_EMPLOYED.values[0], color='firebrick')
                return fig
            st.pyplot(plot())


        with pos_3:
            # @st.cache_resource()
            def plot():                
                fig, ax = plt.subplots()
                splot = sns.scatterplot(x=dash_df['AMT_INCOME_TOTAL'], y=dash_df['DAYS_EMPLOYED'])
                splot.set(xscale="log")
                plt.scatter(x=dash_df[dash_df.index==id].AMT_INCOME_TOTAL.values[0], 
                            y=dash_df[dash_df.index==id].DAYS_EMPLOYED.values[0], color='firebrick')
                return fig
            st.pyplot(plot())


        with pos_1:
            # @st.cache_resource()
            def plot():                
                fig, ax = plt.subplots()
                splot = sns.scatterplot(x=dash_df['AMT_INCOME_TOTAL'], y=dash_df['DAYS_BIRTH'], 
                                        hue=dash_df['FLAG_OWN_CAR'])
                splot.set(xscale="log")
                plt.scatter(x=dash_df[dash_df.index==id].AMT_INCOME_TOTAL.values[0], 
                            y=dash_df[dash_df.index==id].DAYS_BIRTH.values[0], color='firebrick')
                return fig
            st.pyplot(plot())

        with pos_2:
            # @st.cache_resource()
            def plot():                
                fig, ax = plt.subplots()
                splot = sns.scatterplot(x=dash_df['AMT_INCOME_TOTAL'], y=dash_df['DAYS_BIRTH'], 
                                        hue=dash_df['NAME_EDUCATION_TYPE'])
                splot.set(xscale="log")
                plt.scatter(x=dash_df[dash_df.index==id].AMT_INCOME_TOTAL.values[0], 
                            y=dash_df[dash_df.index==id].DAYS_BIRTH.values[0], color='firebrick')
                return fig
            st.pyplot(plot())

        with pos_3:
            # @st.cache_resource()
            def plot():                
                fig, ax = plt.subplots()
                splot = sns.scatterplot(x=dash_df['AMT_INCOME_TOTAL'], y=dash_df['NAME_EDUCATION_TYPE'])
                splot.set(xscale="log")
                plt.scatter(x=dash_df[dash_df.index==id].AMT_INCOME_TOTAL.values[0], 
                            y=dash_df[dash_df.index==id].NAME_EDUCATION_TYPE.values[0], color='firebrick')
                return fig
            st.pyplot(plot())



    chk_loan = st.checkbox("Show Customer Loan Informations ?")

    if chk_loan:

        c1, c2,  = st.columns(2)
        with st.container(): 
            c1.write("**Loan Type** : " + str(dash_df[dash_df.index==id].NAME_CONTRACT_TYPE.values[0]))
            c2.write('**Loan amount** : ' + str(dash_df[dash_df.index==id].AMT_CREDIT.values[0]) + ' USD')

        pos_1, pos_2, pos_3 = st.columns(3)
        
        with pos_1:
            # @st.cache_data(hash_funcs={matplotlib.figure.Figure: lambda _: None})
            def plot():                
                fig, ax = plt.subplots()
                sns.kdeplot(data=dash_df, x='AMT_CREDIT', label = 'Loan amount', log_scale=True, hue='NAME_CONTRACT_TYPE')
                plt.axvline(x=dash_df[dash_df.index==id].AMT_CREDIT.values[0], ymax=0.95, color='firebrick', ls='--')
                return fig
            st.write(plot())

        with pos_2:
            # @st.cache_data(hash_funcs={matplotlib.figure.Figure: lambda _: None})
            def plot():                
                fig, ax = plt.subplots()
                splot = sns.scatterplot(y=dash_df['AMT_INCOME_TOTAL'], x=dash_df['AMT_CREDIT'])
                splot.set(xscale="log", yscale="log")
                plt.scatter(y=dash_df[dash_df.index==id].AMT_INCOME_TOTAL.values[0], 
                            x=dash_df[dash_df.index==id].AMT_CREDIT.values[0], color='firebrick')
                return fig
            st.write(plot())

        with pos_3:
            # @st.cache_data(hash_funcs={matplotlib.figure.Figure: lambda _: None})
            def plot():                
                fig, ax = plt.subplots()
                sns.kdeplot(data=dash_df, x='AMT_CREDIT', log_scale=True, hue='FLAG_OWN_REALTY')
                plt.axvline(x=dash_df[dash_df.index==id].AMT_CREDIT.values[0], ymax=0.95, color='firebrick', ls='--')
                return fig
            st.write(plot())

        with pos_1:
            # @st.cache_data(hash_funcs={matplotlib.figure.Figure: lambda _: None})
            def plot():                
                fig, ax = plt.subplots()
                splot = sns.scatterplot(y=dash_df['DAYS_BIRTH'], x=dash_df['AMT_CREDIT'])
                splot.set(xscale="log",)
                plt.scatter(y=dash_df[dash_df.index==id].DAYS_BIRTH.values[0], 
                            x=dash_df[dash_df.index==id].AMT_CREDIT.values[0], color='firebrick')
                return fig
            st.write(plot())

        with pos_2:
            # @st.cache_data(hash_funcs={matplotlib.figure.Figure: lambda _: None})
            def plot():                
                fig, ax = plt.subplots()
                splot = sns.scatterplot(y=dash_df['DAYS_EMPLOYED'], x=dash_df['AMT_CREDIT'])
                splot.set(xscale="log",)
                plt.scatter(y=dash_df[dash_df.index==id].DAYS_EMPLOYED.values[0], 
                            x=dash_df[dash_df.index==id].AMT_CREDIT.values[0], color='firebrick')
                return fig
            st.write(plot())


###########################################################################################################################    
###########################################################################################################################
with intrepretation:
    st.markdown('***')    
    original_title = '<p style="font-size: 25px;text-align: left; color:Blue;"> Interpretation (Features Importance): </p>'
    st.markdown(original_title, unsafe_allow_html=True)

    # here is the original index from 0 to 47255
    X_similar = X_dashboard.reset_index()
    idx_client=X_similar[X_similar['SK_ID_CURR']==id].index
    X_similar = X_similar.drop(columns='SK_ID_CURR')
    X_similar['proba'] = model.predict_proba(X_similar)[:, 1]

    chk_global = st.checkbox("Show Global interpretation ?")

    if chk_global:

        feature_number = st.slider('How Many features to display?', 2, 20, 8, 2)
    
        #explain the model's predictions using SHAP
        #(same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
        # @st.cache
        # def load_data_1():
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(X_dashboard)[0]
        exp_value=explainer.expected_value[0]
        #     return explainer, shap_values, exp_value

        # explainer, shap_values, exp_value = load_data_1()
        

        
        #st.set_option('deprecation.showPyplotGlobalUse', False)

        ### Global interortation ###
        original_title = '<p style="font-size: 20px;text-align: left; color:green;"> Global Interpretation: </p>'
        st.markdown(original_title, unsafe_allow_html=True)

        #summary_plot
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_dashboard.astype("float"), max_display=feature_number )
        st.pyplot(fig)

        #summary_plot_bar
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_dashboard, plot_type="bar", max_display=feature_number )
        st.pyplot(fig)


        ##################################
        # features description select list 
        original_title = '<p style="font-size: 20px;text-align: left; color:green;"> please select feature to show description: </p>'
        st.markdown(original_title, unsafe_allow_html=True)

        feature_list = description_df.Row.unique().tolist()
        feat = st.selectbox('Features', options=feature_list)
        description = description_df[description_df['Row']==feat].Description.tolist()

        st.markdown(description)
        st.markdown("***")
        ##################################

        # shap features plot
        vals= np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(X_dashboard.columns,vals)),columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
        shap_features_list = feature_importance.col_name.head(feature_number).tolist()

        options = st.multiselect(
        'Please select feature to plot', shap_features_list, shap_features_list[:2])

        pos_1, pos_2, pos_3 = st.columns(3)
        
        for i in range(len(options)):
            pos = pos_3 if (i-3*(i//3))==2  else pos_2 if (i-3*(i//3))==1 else pos_1
            
            with pos:
                fig, ax = plt.subplots()
                sns.kdeplot(X_similar[options[i]][X_similar['proba']>th],color='red', label='Target=1')
                sns.kdeplot(X_similar[options[i]][X_similar['proba']<th],color='green', label='Target=0')
                plt.axvline(x=X_similar.loc[idx_client, options[i]].values[0], ymax=0.95, color='black', ls='--', label='customer')
                plt.legend()
                st.pyplot(fig)


        # st.write('You selected:', X_similar.loc[idx_client, options[0]].values[0])

        ##################################

        chk_local = st.checkbox("Show local interpretation ?")

        if chk_local:

            ### Local interortation ###
            original_title = '<p style="font-size: 20px;text-align: left; color:green;"> Local Interpretation: </p>'
            st.markdown(original_title, unsafe_allow_html=True)

            # function to display shap values
            def st_shap(plot, height=None):
                shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                components.html(shap_html, height=height)
            
            #force_plot
            st_shap(shap.force_plot(exp_value, shap_values[idx_client], features = X_dashboard.iloc[idx_client], feature_names=X_dashboard.columns, figsize=(12,5)))

            #waterfall_plot
            fig, ax = plt.subplots()
            shap.waterfall_plot(shap.Explanation(values=shap_values[idx_client.values[0]], 
            base_values=exp_value, data=X_dashboard.iloc[idx_client.values[0]],  
            feature_names=X_dashboard.columns.tolist()))
            st.pyplot(fig)


###########################################################################################################################
###########################################################################################################################    
with comparaison:
    st.markdown('***')    
    original_title = '<p style="font-size: 25px;text-align: left; color:Blue;"> Comparaison (Similar customer): </p>'
    st.markdown(original_title, unsafe_allow_html=True)


    X_display = dash_df.reset_index()

    # X_similar = X_similar.drop(columns='SK_ID_CURR')
    # X_similar['proba'] = model.predict_proba(X_similar)[:, 1]
    X_neigh = pd.DataFrame(np.matrix([np.ones(X_similar.shape[0]), X_similar['proba'].values]).T)

    #Similar customer files display
    chk_voisins_1 = st.checkbox("Show similar customer Files ?")

    if chk_voisins_1:
        neigh= NearestNeighbors(n_neighbors=10)
        neigh.fit(X_similar)
        idx_neigh_1=neigh.kneighbors(X_similar.loc[idx_client].values,return_distance=False)[0]
        idx_neigh_1=np.sort(idx_neigh_1)
        neigh_display_1=X_display.loc[idx_neigh_1]
        neigh_display_1['proba'] = X_similar['proba'].loc[idx_neigh_1]
        st.dataframe(neigh_display_1)

    chk_voisins_2 = st.checkbox("Show similar customer Scores ?")

    if chk_voisins_2:
        neigh= NearestNeighbors(n_neighbors=10)
        neigh.fit(X_neigh)
        idx_neigh=neigh.kneighbors(X_neigh.loc[idx_client].values,return_distance=False)[0]
        idx_neigh=np.sort(idx_neigh)
        neigh_display=X_display.loc[idx_neigh]
        neigh_display['proba'] = X_similar['proba'].loc[idx_neigh]
        st.dataframe(neigh_display)


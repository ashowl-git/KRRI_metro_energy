
import glob 
import os
import sys, subprocess
from subprocess import Popen, PIPE
import numpy as np
import pandas as pd

import streamlit as st
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go 
import chart_studio.plotly as py
import cufflinks as cf
import math
from datetime import datetime
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# 사이킷런 라이브러리 불러오기 _ 통계, 학습 테스트세트 분리, 선형회귀등
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error
from PIL import Image
from pyparsing import empty

pd.set_option('display.float_format', '{:,.2f}'.format)


# # hide the hamburger menu? hidden or visible
hide_menu_style = """
        <style>
        #MainMenu {visibility: visible;}
        footer {visibility: visible;}
        footer:after {content:'Copyright 2023. 한국철도기술연구원. All rights reserved.';
        display:block;
        opsition:relatiive;
        color:orange; #tomato 
        padding:5px;
        top:100px;}

        </style>
        """

st.set_page_config(layout="wide", page_title="KRRI_metro_Energy")
st.markdown(hide_menu_style, unsafe_allow_html=True) # hide the hamburger menu?

tab0, tab1, tab2 = st.tabs(['프로그램 개요 및 사용자 메뉴얼','패시브-액티브 기술을 적용한 2차에너지 성능분석', 'ZEB 취득을 위한 신재생 에너지 제안'])

#  필요한 데이터 불러오기
@st.cache_data
def get_sun_data(path, sheet):
    data = pd.read_excel(path, sheet_name=sheet)
    return data

DF1 = get_sun_data('data/DB.xlsx', '01_sun')
DF2 = get_sun_data('data/DB.xlsx', '02_sun2')
DF3 = get_sun_data('data/DB.xlsx', '03_clearsky')
DF5 = get_sun_data('data/DB.xlsx', '04_renewable')
DF6 = get_sun_data('data/DB.xlsx', '05_zero')
DF7 = get_sun_data('data/DB.xlsx', '06_price')


# DF1 = pd.read_excel('data/DB.xlsx', sheet_name='01_sun')  #일사량
# DF2 = pd.read_excel('data/DB.xlsx', sheet_name='02_sun2') #경사일사량
# DF3 = pd.read_excel('data/DB.xlsx', sheet_name='03_clearsky') # 맑은날
# DF5 = pd.read_excel('data/DB.xlsx', sheet_name='04_renewable') # 신재생
# DF6 = pd.read_excel('data/DB.xlsx', sheet_name='05_zero') # 신재생
# DF7 = pd.read_excel('data/DB.xlsx', sheet_name='06_price') # 가격

with tab0 : 
    empty1, con1, empty2 = st.columns([0.1, 1.0, 0.1])

    with empty1 :
        empty()

    with con1 : 
        st.subheader('제로에너지 철도역사 건설 전략수립 의사결정 지원 프로그램')
        st.subheader('')
        
        st.markdown("[:white_check_mark:사용자매뉴얼 다운로드](https://drive.google.com/uc?export=download&id=1aQSsvtjptAJpWD1y1hM8zrKT25aUQgQt)")

        st.markdown("![](https://drive.google.com/uc?export=download&id=1M1qtS3qPJ3qxzYyQivBRjHGpYpIU7C4l)")
        st.markdown("![](https://drive.google.com/uc?export=download&id=1uvrANk6R_HWrh7Xnpvk2VTu2dCjFgiiN)")
        st.markdown("![](https://drive.google.com/uc?export=download&id=117CYlBbmw4g1LRMX-T1Tid0mQ_pWLEgs)")
        st.markdown("![](https://drive.google.com/uc?export=download&id=1QnHSI5D96KN1C5FZVZD1oDE5lTdj8CG_)")
        st.markdown("![](https://drive.google.com/uc?export=download&id=1dqo2H-JxEgOnrkIl976aj7EBX6r7c5Eb)")


        #@st.cache_data
        #def show_image(path):
        #    img = Image.open(path)
  
        #img4 = show_image('data/사용자 메뉴얼_2.jpg')
        #img5 = show_image('data/사용자 메뉴얼_3.jpg')
        # img6 = show_image('data/사용자 메뉴얼_4.jpg')
        # img7 = show_image('data/사용자 메뉴얼_5.jpg')
        # img8 = show_image('data/사용자 메뉴얼_6.jpg')
        # img9 = show_image('data/사용자 메뉴얼_7.jpg')
        # img10 = show_image('data/사용자 메뉴얼_8.jpg')
        # img11 = show_image('data/사용자 메뉴얼_9.jpg')
        # img12 = show_image('data/사용자 메뉴얼_10.jpg')
        # img13 = show_image('data/사용자 메뉴얼_11.jpg')
        # img14 = show_image('data/사용자 메뉴얼_12.jpg')
        # img15 = show_image('data/사용자 메뉴얼_13.jpg')
        # img16 = show_image('data/사용자 메뉴얼_14.jpg')
        # img17 = show_image('data/사용자 메뉴얼_15.jpg')

        # img4 = Image.open('data/사용자 메뉴얼_2.jpg')
        # st.image(img4)
        # img5 = Image.open('data/사용자 메뉴얼_3.jpg')
        # st.image(img5)
        # img6 = Image.open('data/사용자 메뉴얼_4.jpg')
        # st.image(img6)
        # img7 = Image.open('data/사용자 메뉴얼_5.jpg')
        # st.image(img7)
        # img8 = Image.open('data/사용자 메뉴얼_6.jpg')
        # st.image(img8)
        # img9 = Image.open('data/사용자 메뉴얼_7.jpg')
        # st.image(img9)
        # img10 = Image.open('data/사용자 메뉴얼_8.jpg')
        # st.image(img10)
        # img11 = Image.open('data/사용자 메뉴얼_9.jpg')
        # st.image(img11)
        # img12 = Image.open('data/사용자 메뉴얼_10.jpg')
        # st.image(img12)
        # img13 = Image.open('data/사용자 메뉴얼_11.jpg')
        # st.image(img13)
        # img14 = Image.open('data/사용자 메뉴얼_12.jpg')
        # st.image(img14)
        # img15 = Image.open('data/사용자 메뉴얼_13.jpg')
        # st.image(img15)
        # img16 = Image.open('data/사용자 메뉴얼_14.jpg')
        # st.image(img16)
        # img17 = Image.open('data/사용자 메뉴얼_15.jpg')
        # st.image(img17)

    with empty2 :
        empty()


#_________________________________________________________________________________________________________________________________________________

# @st.cache_data
# def get_raw_data(path):
#     data = pd.read_excel(path)
#     return data
# df_raw = get_raw_data('data/metro_sim_month.xlsx')

with tab1 : 

    st.subheader('■ 패시브-액티브 기술을 적용한 2차에너지 성능분석 ')
    st.markdown("###### - 예측 모델 학습데이터 입력 및 개선전후 기술요소 입력을 통한 에너지성능 확인")

    st.caption('--------- ', unsafe_allow_html=False)
    st.markdown("#### 1. 학습 데이터 업로드 및 모델성능 확인")
    st.caption('         ', unsafe_allow_html=False)

    st.markdown("##### ● 에너지 데이터 학습")
    # 학습파일 불러오기

    df_raw = pd.read_excel('data/metro_sim_month.xlsx')

    basecon0a, basecon1a = st.columns([0.01, 4])
    with basecon1a : 
    
        box_학습데이터_업로드 = st.checkbox('학습 데이터 업로드(필요시 체크, 미 업로드 시 기본모델로 시뮬레이션 수행)')
        if box_학습데이터_업로드 : 
            st.subheader(' 학습데이터 직접 업로드')
            st.caption('(업로드 하지 않아도 기본값으로 작동합니다)', unsafe_allow_html=False)
            # 학습할 파일을 직접 업로드 하고 싶을때
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                df_raw = pd.read_excel(uploaded_file)
                st.write(df_raw)


    # df_raw.columns
    df_raw2 = df_raw.copy()

    # Alt 용 독립변수 데이터셋 컬럼명 수정
    df_raw2 = df_raw2.rename(columns={
        'ACH50':'ACH50_2',
        'Lighting_power_density_':'Lighting_power_density__2',
        'Chiller_COP':'Chiller_COP_2',
        'Pump_efficiency':'Pump_efficiency_2',
        'Fan_total_efficiency':'Fan_total_efficiency_2',
        'heat_recover_effectiveness':'heat_recover_effectiveness_2',
        'AHU_economiser':'AHU_economiser_2',
        'Occupied_floor_area':'Occupied_floor_area_2',
        'Floor':'Floor_2',
        'Basement':'Basement_2',
        'Ground':'Ground_2',
        })


    # 독립변수컬럼 리스트
    lm_features =['ACH50', 'Lighting_power_density_', 'Chiller_COP', 'Pump_efficiency',
        'Fan_total_efficiency', 'heat_recover_effectiveness', 'AHU_economiser',
        'Occupied_floor_area', 'Floor', 'Basement', 'Ground',]

    # Alt 용 독립변수 데이터셋 컬럼명 리스트
    lm_features2 =['ACH50_2', 'Lighting_power_density__2', 'Chiller_COP_2', 'Pump_efficiency_2',
        'Fan_total_efficiency_2', 'heat_recover_effectiveness_2', 'AHU_economiser_2',
        'Occupied_floor_area_2', 'Floor_2', 'Basement_2', 'Ground_2',]

    # 종속변수들을 드랍시키고 독립변수 컬럼만 X_data에 저장
    X_data = df_raw[lm_features]
    X_data2 = df_raw2[lm_features2]


    # X_data 들을 실수로 변경
    X_data = X_data.astype('float')
    X_data2 = X_data2.astype('float')

    # 독립변수들을 드랍시키고 종속변수 컬럼만 Y_data에 저장
    Y_data = df_raw.drop(df_raw[lm_features], axis=1, inplace=False)
    Y_data2 = df_raw2.drop(df_raw2[lm_features2], axis=1, inplace=False)
    lm_result_features = Y_data.columns.tolist()
    lm_result_features2 = Y_data2.columns.tolist()


    # 학습데이터에서 일부를 분리하여 테스트세트를 만들어 모델을 평가 학습8:테스트2
    X_train, X_test, y_train, y_test = train_test_split(
    X_data, Y_data , 
    test_size=0.2, 
    random_state=150)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_data2, Y_data2 , 
    test_size=0.2, 
    random_state=150)

    # 학습 모듈 인스턴스 생성
    lr = LinearRegression() 
    lr2 = LinearRegression() 

    # 인스턴스 모듈에 학습시키기
    lr.fit(X_train, y_train)
    lr2.fit(X_train2, y_train2)

    # 테스트 세트로 예측해보고 예측결과를 평가하기
    y_preds = lr.predict(X_test)
    y_preds2 = lr2.predict(X_test2)

    mse = mean_squared_error(y_test, y_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_preds)
    mape = mean_absolute_percentage_error(y_test, y_preds)

    # Mean Squared Logarithmic Error cannot be used when targets contain negative values.
    # msle = mean_squared_log_error(y_test, y_preds)
    # rmsle = np.sqrt(msle)

    print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
    print('MAE : {0:.3f}, MAPE : {1:.3f}'.format(mae, mape))
    # print('MSLE : {0:.3f}, RMSLE : {1:.3f}'.format(msle, rmsle))

    print('Variance score(r2_score) : {0:.3f}'.format(r2_score(y_test, y_preds)))
    r2 = r2_score(y_test, y_preds)

    st.caption('         ', unsafe_allow_html=False)
    st.markdown("##### ● 예측모델 성능 확인")
    
    basecon0b, basecon1b = st.columns([0.01, 4])
    with basecon1b :

        col1, col2, col3, = st.columns(3)
        col1.metric(label='Variance score(r2_score)', value = np.round(r2, 3))
    # col2.metric(label='mean_squared_error', value = np.round(mse, 3))
    #col2.metric(label='Root mean squared error(RMSE)', value = np.round(rmse, 3))
    # col4.metric(label='mean_absolute_error', value = np.round(mae, 3))
    #col3.metric(label='Mean absolute percentage error(MAPE)', value = np.round(mape, 3))
    
    # print('절편값:',lr.intercept_)
    # print('회귀계수값:',np.round(lr.coef_, 1))


    # 회귀계수를 테이블로 만들어 보기 1 전치하여 세로로 보기 (ipynb 확인용)
    coeff = pd.DataFrame(np.round(lr.coef_,2), columns=lm_features).T
    coeff2 = pd.DataFrame(np.round(lr.coef_,2), columns=lm_features2).T

    coeff.columns = lm_result_features
    coeff2.columns = lm_result_features2

    # st.subheader('LinearRegression 회귀계수')
    # st.caption('--------', unsafe_allow_html=False)
    # coeff
    # coeff2

    # Sidebar
    # Header of Specify Input Parameters

    #개선전후 공통정보
    st.caption('--------- ', unsafe_allow_html=False)
    st.markdown("#### 2. 철도역사 기본정보 및 기술정보 입력")
    st.caption('         ', unsafe_allow_html=False)
    st.markdown("##### ● 철도역사 기본정보 입력")
    
    basecon0, basecon1, basecon2, basecon3, basecon4 = st.columns([0.01, 0.5, 0.5, 0.5, 0.5])
    with basecon1 : 
        Ground_base = st.select_slider('① 지하층유무', options=[0, 1])
    with basecon2 :   
        Basement_base = st.select_slider('② 지상층유무', options=[0, 1])
    with basecon3 :     
        Floor_base = st.select_slider('③ 전체층규모', options=[1,2,3,4,5])
    with basecon4 :   
        Occupied_floor_area_base = st.number_input('④ 연면적(㎡)', 1000, 100000, 6000)

    # base 모델 streamlit 인풋
    st.caption(' ', unsafe_allow_html=False)
    #st.caption('--------', unsafe_allow_html=False)
    #st.subheader('■ 현 철도역사 건물 정보입력')
    st.markdown("##### ● 현 철도역사 건물 정보입력(개선전)")
        
    def user_input_features():
        con0, con1, con2, con3, con4 = st.columns([0.01, 0.5, 0.5, 0.5, 0.5])
        # ACH50 = st.sidebar.slider('ACH50', X_data.ACH50.min(), X_data.ACH50.max(), X_data.ACH50.mean())
        with con1 : 
            #Ground = st.select_slider('지하층유무_개선전', options=[0, 1])
            Ground = Ground_base
            ACH50 = st.number_input('① 침기율(ACH/50pa)_개선전', 0.0, 50.0, 5.0)
            Pump_efficiency = st.number_input('⑤ 펌프효율_개선전', 0.0, 1.0, 0.8)
            
        with con2 : 
            #Basement = st.select_slider('지상층유무_개선전', options=[0, 1])
            Basement = Basement_base
            Chiller_COP = st.number_input('② 냉동기(COP)_개선전', 4, 9, 7)
            heat_recover_effectiveness = st.number_input('⑥ 전열교환효율_개선전', 0.0, 1.0, 0.7)
       
        with con3 : 
            #Floor = st.select_slider('전체층규모_개선전', options=[1,2,3,4,5])
            Floor = Floor_base
            Fan_total_efficiency = st.number_input('③ 팬효율_개선전', 0.0, 1.0, 0.8)
            Lighting_power_density_ = st.number_input('⑦ 조명밀도(W/㎡)_개선전', 3, 20, 7)
      
        with con4 :
            #Occupied_floor_area = st.number_input('연면적(㎡)_개선전', 1000, 100000, 6000)
            Occupied_floor_area = Occupied_floor_area_base
            AHU_economiser = st.select_slider('④ AHU_이코노마이저 적용유무_개선전', options=[0, 1])     
            
            data = {'ACH50': ACH50,
                    'Lighting_power_density_': Lighting_power_density_,
                    'Chiller_COP': Chiller_COP,
                    'Pump_efficiency': Pump_efficiency,
                    'Fan_total_efficiency': Fan_total_efficiency,
                    'heat_recover_effectiveness': heat_recover_effectiveness,
                    'AHU_economiser': AHU_economiser,
                    'Occupied_floor_area': Occupied_floor_area,
                    'Floor': Floor,
                    'Basement': Basement,
                    'Ground': Ground,}
        features = pd.DataFrame(data, index=[0])
        return features
    df_input = user_input_features()
    result = lr.predict(df_input)

    # ALT 모델 streamlit 인풋
     
       
    #st.caption('--------', unsafe_allow_html=False)
    #st.subheader('■ 개선하고자 하는 철도역사 건물 정보입력')
    st.caption('         ', unsafe_allow_html=False)
    st.markdown("##### ● 개선하고자 하는 철도역사 건물 정보입력(개선후)")

    def user_input_features2():
        con0, con1, con2, con3, con4 = st.columns([0.01, 0.5, 0.5, 0.5, 0.5])
            # ACH50 = st.sidebar.slider('ACH50', X_data.ACH50.min(), X_data.ACH50.max(), X_data.ACH50.mean())
        with con1 : 
            #Ground_2 = st.select_slider('지하층유무_개선후', options=[0, 1]) 
            Ground_2 = Ground_base
            ACH50_2 = st.number_input('① 침기율(ACH/50pa)_개선후', 0.0, 50.0, 5.0)
            Pump_efficiency_2 = st.number_input('⑤ 펌프효율_개선후', 0.0, 1.0, 0.8)
            
        with con2 : 
            #Basement_2 = st.select_slider('지상층유무_개선후', options=[0, 1])
            Basement_2 = Basement_base
            Chiller_COP_2 = st.number_input('② 냉동기(COP)_개선후', 4, 9, 7)
            heat_recover_effectiveness_2 = st.number_input('⑥ 전열교환효율_개선후', 0.0, 1.0, 0.7)
            
        with con3 :  
            #Floor_2 = st.select_slider('전체층규모_개선후', options=[1,2,3,4,5])   
            Floor_2 = Floor_base
            Fan_total_efficiency_2 = st.number_input('③ 팬효율_개선후', 0.0, 1.0, 0.8)
            Lighting_power_density__2 = st.number_input('⑦ 조명밀도(W/㎡)_개선후', 3, 20, 7)
            
            
        with con4 :   
            #Occupied_floor_area_2 = st.number_input('연면적(㎡)_개선후', 1000, 100000, 6000)
            Occupied_floor_area_2 = Occupied_floor_area_base
            AHU_economiser_2 = st.select_slider('④ AHU_이코노마이저 적용유무_개선후', options=[0, 1])
            
            data2 = {'ACH50_2': ACH50_2,
                'Lighting_power_density__2': Lighting_power_density__2,
                'Chiller_COP_2': Chiller_COP_2,
                'Pump_efficiency_2': Pump_efficiency_2,
                'Fan_total_efficiency_2': Fan_total_efficiency_2,
                'heat_recover_effectiveness_2': heat_recover_effectiveness_2,
                'AHU_economiser_2': AHU_economiser_2,
                'Occupied_floor_area_2': Occupied_floor_area_2,
                'Floor_2': Floor_2,
                'Basement_2': Basement_2,
                'Ground_2': Ground_2,}
                    
        features2 = pd.DataFrame(data2, index=[0])
        return features2

    df2_input = user_input_features2()

    result2 = lr2.predict(df2_input)
    
    #######################################
    # st.subheader('에너지 사용량 예측값')
    # st.caption('좌측의 변수항목 슬라이더 조정 ', unsafe_allow_html=False)
    # st.caption('--------- ', unsafe_allow_html=False)
    #######################################

    # 예측된 결과를 데이터 프레임으로 만들어 보기
    df_result = pd.DataFrame(result, columns=lm_result_features).T.rename(columns={0:'kW'})
    df_result2 = pd.DataFrame(result2, columns=lm_result_features2).T.rename(columns={0:'kW'})


    df_result['Alt'] = '개선전'
    df_result2['Alt'] = '개선후'

    df_result['kW/m2'] = df_result['kW'] / df_input['Occupied_floor_area'][0]
    df_result2['kW/m2'] = df_result2['kW'] / df2_input['Occupied_floor_area_2'][0]


    # df_result
    # df_result2

    df_result.reset_index(inplace=True)
    df_result2.reset_index(inplace=True)

    # df_result.rename(columns={'index':'BASE_index'})
    # df_result2.rename(columns={'index':'BASE_index2'})
    # 숫자만 추출해서 행 만들기 
    # 숫자+'호' 문자열 포함한 행 추출해서 행 만들기 df['floor'] = df['addr'].str.extract(r'(\d+호)')

    # 숫자만 추출해서 Month 행 만들기
    df_result['Month'] = df_result['index'].str.extract(r'(\d+)')
    df_result2['Month'] = df_result2['index'].str.extract(r'(\d+)')
    df_result['index']  =df_result['index'].str.slice(0,-3)
    df_result2['index'] = df_result2['index'].str.slice(0,-3)

    df_concat = pd.concat([df_result,df_result2])



    # 추세에 따라 음수값이 나오는것은 0으로 수정
    cond1 = df_concat['kW'] < 0
    df_concat.loc[cond1,'kW'] = 0


    df_concat = df_concat.reset_index(drop=True)
    df_concat = df_concat.round(2)

    df_concat_연간전체 = df_concat.groupby('Alt').agg(연간전기사용량_전체 = ('kW', 'sum'), 단위면적당_연간전기사용량_전체 = ('kW/m2', 'sum'))
    df_concat_월간전체 = df_concat.groupby(['Alt','Month']).agg( 월간전기사용량_전체 = ('kW', 'sum'), 단위면적당_월간전기사용량_전체 = ('kW/m2', 'sum'))
    df_concat_연간원별 = df_concat.groupby('index').agg(연간전기사용량_원별 = ('kW', 'sum'), 단위면적당_연간전기사용량_원별 = ('kW/m2', 'sum'))
    df_concat_월간원별 = df_concat.groupby(['index','Month']).agg(연간전기사용량_원별 = ('kW', 'sum'), 단위면적당_연간전기사용량_원별 = ('kW/m2', 'sum'))

    df_concat_연간전체 = df_concat_연간전체.reset_index()
    df_concat_월간전체 = df_concat_월간전체.reset_index()
    df_concat_연간원별 = df_concat_연간원별.reset_index()
    df_concat_월간원별 = df_concat_월간원별.reset_index()
    
    # df_concat_월간원별.plot.bar()
    
    # 예측값을 데이터 프레임으로 만들어본것을 그래프로 그려보기
    st.caption('--------- ', unsafe_allow_html=False)
    st.markdown("#### 3. 개선전후 에너지성능 비교")
    st.caption('         ', unsafe_allow_html=False)
    st.markdown("##### ● 개선전후 연간 2차 에너지 사용량(그래프)")
          
    con0, con1, con2, con3, con4 = st.columns([0.01, 0.5, 0.5, 0.5, 0.5])
    with con1 : 
            
        fig = px.box(
            df_concat, x='index', y='kW', 
            title='① 개선전후 원별비교 (BOXplot)', 
            hover_data=['kW'], 
            color='Alt' )
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(barmode='group') #alt별 구분
        # fig
        st.plotly_chart(fig, use_container_width=True)
        
    with con2 : 
            
        fig = px.bar(df_concat, x='index', y='kW', title='② 개선전후 원별비교', hover_data=['kW'], color='Alt' )
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(barmode='group') #alt별 구분
        # fig
        st.plotly_chart(fig, use_container_width=True)

        
    with con3 :
        fig = px.bar(
        df_concat_연간전체, x='Alt', y='연간전기사용량_전체', 
        title='③ 개선전후 에너지사용량', 
        hover_data=['연간전기사용량_전체'], 
        color='Alt' )
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(barmode='group') #alt별 구분
        # fig
        st.plotly_chart(fig, use_container_width=True)

    with con4 : 
        fig = px.bar(
        df_concat_연간전체, x='Alt', y='단위면적당_연간전기사용량_전체', 
        title='④ 개선전후 단위면적당 에너지사용량', 
        hover_data=['단위면적당_연간전기사용량_전체'], 
        color='Alt' )
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(barmode='group') #alt별 구분
        # fig
        st.plotly_chart(fig, use_container_width=True)

    #연간에너지표 정리 
    df_concat_월간전체_BASE = df_concat_월간전체.loc[(df_concat_월간전체['Alt'] == '개선전')]
    df_concat_월간전체_ALT = df_concat_월간전체.loc[(df_concat_월간전체['Alt'] == '개선후')]


    box_연간에너지_표 = st.checkbox('개선전후 연간 2차 에너지 사용량 표')
    if box_연간에너지_표 : 
        #st.caption('--------- ', unsafe_allow_html=False)
        st.markdown("##### ● 개선전후 연간 2차 에너지 사용량(표)")
        #st.subheader('■ 개선전후 연간 2차 에너지 사용량 표')
        con2, con3, con4, con5 = st.columns([0.01, 4/3, 4/3, 4/3])
        with con3 :
            st.markdown("###### 연간 전체 에너지 사용량")
            df_concat_연간전체
        with con4 :
            st.markdown("###### 월별에너지 사용량_개선전")
            df_concat_월간전체_BASE
        with con5 :
            st.markdown("###### 월별에너지 사용량_개선후")
            df_concat_월간전체_ALT


    #st.caption('--------- ', unsafe_allow_html=False)
    #st.subheader('■ 개선전후 월별 2차 에너지 사용량 그래프')
    st.markdown("##### ● 개선전후 월별 2차 에너지 사용량(그래프)")
        

    con4, con5, con6, con7 = st.columns([0.01, 4/3, 4/3, 4/3])

    with con5 : 
        # 예측값을 데이터 프레임으로 만들어본것을 그래프로 그려보기
            
        fig = px.bar(df_concat, x='Month', y='kW', title='① 개선전후 월별비교', hover_data=['index'],color='Alt' )
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(barmode='group') #alt별 구분
        # fig
        st.plotly_chart(fig, use_container_width=True)

    with con6 : 
        fig = px.bar(df_result, x='Month', y='kW', title='② 개선전 월간 원별결과', hover_data=['kW'], color='index' )
        fig.update_xaxes(rangeslider_visible=True)
        # fig.update_layout(barmode='group') #alt별 구분
        # fig
        st.plotly_chart(fig, use_container_width=True)

    with con7 :
        fig = px.bar(df_result2, x='Month', y='kW', title='③ 개선후 월간 원별결과', hover_data=['kW'], color='index' )
        fig.update_xaxes(rangeslider_visible=True)
        # fig.update_layout(barmode='group') #alt별 구분
        # fig
        st.plotly_chart(fig, use_container_width=True)

    df_concat_alt_연간원별 = df_concat.groupby(['Alt', 'index']).agg(연간전기사용량_원별 = ('kW','sum'), 단위면적당_연간전기사용량_원별 = ('kW/m2', 'sum'))
    df_concat_alt_연간원별 = df_concat_alt_연간원별.reset_index()
    df_concat_alt_연간원별_개선전 = df_concat_alt_연간원별.loc[(df_concat_alt_연간원별['Alt'] == '개선전')]
    df_concat_alt_연간원별_개선후 = df_concat_alt_연간원별.loc[(df_concat_alt_연간원별['Alt'] == '개선후')]

    df_concat_alt_월간원별 = df_concat.groupby(['Alt', 'index', 'Month']).agg(월간전기사용량_원별 = ('kW','sum'), 단위면적당_월간전기사용량_원별 = ('kW/m2', 'sum'))
    df_concat_alt_월간원별 = df_concat_alt_월간원별.reset_index()
    
    df_concat_alt_월간원별_개선전 = df_concat_alt_월간원별.loc[(df_concat_alt_월간원별['Alt'] == '개선전')]
    df_concat_alt_월간원별_개선후 = df_concat_alt_월간원별.loc[(df_concat_alt_월간원별['Alt'] == '개선후')]

    df_concat_alt_월간원별_개선전_Cooling = df_concat_alt_월간원별_개선전.loc[(df_concat_alt_월간원별_개선전['index'] == 'Cooling')]
    df_concat_alt_월간원별_개선후_Cooling = df_concat_alt_월간원별_개선후.loc[(df_concat_alt_월간원별_개선후['index'] == 'Cooling')]

    df_concat_alt_월간원별_개선전_DHW = df_concat_alt_월간원별_개선전.loc[(df_concat_alt_월간원별_개선전['index'] == 'DHW')]
    df_concat_alt_월간원별_개선후_DHW = df_concat_alt_월간원별_개선후.loc[(df_concat_alt_월간원별_개선후['index'] == 'DHW')]

    df_concat_alt_월간원별_개선전_Fans = df_concat_alt_월간원별_개선전.loc[(df_concat_alt_월간원별_개선전['index'] == 'Fans')]
    df_concat_alt_월간원별_개선후_Fans = df_concat_alt_월간원별_개선후.loc[(df_concat_alt_월간원별_개선후['index'] == 'Fans')]

    df_concat_alt_월간원별_개선전_Heating = df_concat_alt_월간원별_개선전.loc[(df_concat_alt_월간원별_개선전['index'] == 'Heating')]
    df_concat_alt_월간원별_개선후_Heating = df_concat_alt_월간원별_개선후.loc[(df_concat_alt_월간원별_개선후['index'] == 'Heating')]

    df_concat_alt_월간원별_개선전_Lighting = df_concat_alt_월간원별_개선전.loc[(df_concat_alt_월간원별_개선전['index'] == 'Lighting')]
    df_concat_alt_월간원별_개선후_Lighting = df_concat_alt_월간원별_개선후.loc[(df_concat_alt_월간원별_개선후['index'] == 'Lighting')]

    df_concat_alt_월간원별_개선전_Pumps = df_concat_alt_월간원별_개선전.loc[(df_concat_alt_월간원별_개선전['index'] == 'Pumps')]
    df_concat_alt_월간원별_개선후_Pumps = df_concat_alt_월간원별_개선후.loc[(df_concat_alt_월간원별_개선후['index'] == 'Pumps')]

    df_concat_alt_월간원별_개선전_Room_Elec = df_concat_alt_월간원별_개선전.loc[(df_concat_alt_월간원별_개선전['index'] == 'Room_Elec')]
    df_concat_alt_월간원별_개선후_Room_Elec = df_concat_alt_월간원별_개선후.loc[(df_concat_alt_월간원별_개선후['index'] == 'Room_Elec')]

    box_월간에너지_표 = st.checkbox('개선전후 월별 2차 에너지 사용량 표')
    if box_월간에너지_표 : 
        #st.caption('--------- ', unsafe_allow_html=False)
        #st.subheader('■ 개선전후 월별 2차 에너지 사용량 표')
        st.markdown("##### ● 개선전후 월별 2차 에너지 사용량(표)")
        ## 표 원별 정리
        #전체 에너지 사용량_에너지원
        
        con5, con6, con7, con8, con9 = st.columns([0.01, 0.5, 0.5, 0.5, 0.5])
        st.caption('                     ', unsafe_allow_html=False)
        #st.caption('--------- ', unsafe_allow_html=False)
        st.caption('                     ', unsafe_allow_html=False)
        con00, con10, con11, con12, con13 = st.columns([0.01, 0.5, 0.5, 0.5, 0.5])
        with con6 :
            st.markdown("###### 연간 에너지 사용량_개선전")
            df_concat_alt_연간원별_개선전
            st.markdown("###### 연간 에너지 사용량_개선후")
            df_concat_alt_연간원별_개선후
        
        with con7 :
            st.markdown("###### 월간 냉방 에너지 사용량_개선전")
            df_concat_alt_월간원별_개선전_Cooling 
            st.markdown("###### 월간 냉방 에너지 사용량_개선후")
            df_concat_alt_월간원별_개선후_Cooling
        
        with con8 :
            st.markdown("###### 월간 난방 에너지 사용량_개선전")
            df_concat_alt_월간원별_개선전_Heating 
            st.markdown("###### 월간 난방 에너지 사용량_개선후")
            df_concat_alt_월간원별_개선후_Heating
        
        with con9 :
            st.markdown("###### 월간 급탕 에너지 사용량_개선전")
            df_concat_alt_월간원별_개선전_DHW 
            st.markdown("###### 월간 급탕 에너지 사용량_개선후")
            df_concat_alt_월간원별_개선후_DHW
        
        with con10 :
            st.markdown("###### 월간 조명 에너지 사용량_개선전")
            df_concat_alt_월간원별_개선전_Lighting 
            st.markdown("###### 월간 조명 에너지 사용량_개선후")
            df_concat_alt_월간원별_개선후_Lighting
        
        with con11 :
            st.markdown("###### 월간 환기 에너지 사용량_개선전")
            df_concat_alt_월간원별_개선전_Fans 
            st.markdown("###### 월간 환기 에너지 사용량_개선후")
            df_concat_alt_월간원별_개선후_Fans

        with con12 :
            st.markdown("###### 월간 펌프 에너지 사용량_개선전")
            df_concat_alt_월간원별_개선전_Pumps 
            st.markdown("###### 월간 펌프 에너지 사용량_개선후")
            df_concat_alt_월간원별_개선후_Pumps

        with con13 :
            st.markdown("###### 월간 대기전력 에너지 사용량_개선전")
            df_concat_alt_월간원별_개선전_Room_Elec
            st.markdown("###### 월간 대기전력 에너지 사용량_개선후")
            df_concat_alt_월간원별_개선후_Room_Elec
    
#_________________________________________________________________________________________________________________________________________________
    
    
with tab2 :

    st.subheader('■ ZEB 취득을 위한 신재생에너지 제안 ')
    st.markdown("###### - 철도역사 에너지 성능분석 데이터(2차에너지)를 기반으로 건축물 에너지 효율등급 1++등급 및 설정한 제로에너지 등급에 도달할 수 있는 신재생 설치 대안 제시")
    #st.markdown("#### 1. 용어설명")
    #img3 = Image.open('data/3page_calculation_flow.jpg')
    #st.image(img3)

    #계산을 위해 필요한 정보 
    DF4 = df_concat #에너지사용량 예측값 불러오기 

    # base 소요량 합계(Room_Elec제외 합계값 X 보정계수 곱) = hh
    h = DF4.loc[(DF4['Alt'] == '개선전')]
    ss= h[h['index'].str.contains('Room_Elec')].index
    h.drop(ss, inplace=True)
    hh=h['kW/m2'].sum()*2.75
    
    # 개선후 소요량 합계(Room_Elec제외 합계값 X 보정계수 곱) = ii
    i = DF4.loc[(DF4['Alt'] == '개선후')]
    spac= i[i['index'].str.contains('Room_Elec')].index
    i.drop(spac, inplace=True)
    ii=i['kW/m2'].sum()*2.75

    #2차에너지 
    건축물_2차_소요량 = df_concat.groupby(['Alt','Month']).agg( 월간전기사용량 = ('kW', 'sum'), 단위면적당_월간전기사용량 = ('kW/m2', 'sum'))
    건축물_2차_소요량_개선후 = 건축물_2차_소요량.drop(labels='개선전',axis=0)
    #2차에너지 room elec 제외한 값
    i_room_elex_drop_2차 = i.groupby(['Alt','Month']).agg( 월간전기사용량 = ('kW', 'sum'), 단위면적당_월간전기사용량 = ('kW/m2', 'sum'))
    #1차에너지
    i_room_elex_drop_1차 = i_room_elex_drop_2차*2.75
    #연간 1차에너지
    i_room_elex_drop_1차_연 = i_room_elex_drop_1차.groupby(['Alt']).agg( 연간전기사용량 = ('월간전기사용량', 'sum'), 단위면적당_연간전기사용량 = ('단위면적당_월간전기사용량', 'sum'))
    #홈페이지 나타내기
    #st.caption('         ', unsafe_allow_html=False)
    st.caption('--------- ', unsafe_allow_html=False)
    st.markdown("#### 1. 철도역사의 5대에너지(1차 에너지) 산정 과정 ")
    con000, con001,  con003,  con005,  con007 = st.columns([0.01,0.5,0.5,0.5,0.5])
    with con001 : 
        st.markdown("###### ① 에너지 사용량(2차에너지), kW")
        건축물_2차_소요량_개선후

    with con003 : 
        st.markdown("###### ② 건축물 5대 에너지(2차에너지), kW")
        i_room_elex_drop_2차

    with con005 : 
        st.markdown("###### ③ 건축물 5대 에너지(1차에너지), kW")
        i_room_elex_drop_1차

    with con007 : 
        st.markdown("###### ④ 건축물 5대 에너지(연간 1차에너지), kW")
        i_room_elex_drop_1차_연
        
    st.caption('                     ', unsafe_allow_html=False)
    st.caption('--------- ', unsafe_allow_html=False)
    st.markdown("#### 2. 신재생 용량 산정을 위한 기본정보 입력")
    st.caption('                     ', unsafe_allow_html=False)
    st.markdown("##### ● 건축물 기본정보")
    con00, con10, con11, con12, con13 = st.columns([0.01, 0.7, 0.3, 0.7, 0.3])

    with con10 : 
        
        지역명 = ['서울','강릉', '광주', '대관령', '대구', '대전', '목포','부산', '서산', '원주', '인천', '전주', '청주', '추풍령', '춘천', '포항', '흑산도']
        지역 = st.selectbox('① 지역', 지역명)

        #area2 = st.number_input('＊연면적(㎡)', 1000, 100000, 6000)
        area2 = Occupied_floor_area_base
        #st.caption("(전체 연면적을 입력)", unsafe_allow_html=False)
        
        air_ratio = st.number_input('② 공조면적비율(%)', 0, 100, 8)
        st.caption("(전체연면적 대비 외기와 직접 통하지 않으며 냉난방이 공급되는 면적의 비율)", unsafe_allow_html=False)

        area_air = area2*(air_ratio/100) #공조면적
        area_air2 = format(int(area_air),',d')
        kkkk = f'{area_air2}㎡'
        st.markdown(f"##### ▶ 공조면적 : :blue[{kkkk}]")
        #st.subheader('' :blue[kkkk]'')
        #st.caption('                     ', unsafe_allow_html=False)
        #st.caption('--------- ', unsafe_allow_html=False)
        #st.markdown("###### ②지열에너지 기본정보")
        #area4 = st.number_input('＊지열히트펌프공급면적(㎡)', 1000, 100000, 5000)
        #st.caption("(지열히트펌프를 공급하고자 하는 실의 면적 입력)", unsafe_allow_html=False)


    with con11 : 
        empty()    
       
    with con12 : 
        st.markdown("##### ● 제로에너지목표등급 설정")
        제로에너지등급 = st.number_input('① 제로에너지목표등급 입력', 1, 4, 4)
        st.caption("(목표하고자 하는 제로에너지 등급 입력)", unsafe_allow_html=False)

        #st.markdown("###### ③태양광 모듈 1개에 대한 기본정보")
        #LENGTH = st.number_input('＊가로길이 (mm)', 0, 5000, 1000)
        #WIDTH = st.number_input('＊세로길이 (mm)', 0, 5000, 2000)
        #방위별경사각 = ['South_15', 'South_30', 'South_45', 'South_60', 'South_75', 'South_90', 'East_90', 'West_90', 'North_90']
        #경사각도 = st.selectbox('＊방위_경사', 방위별경사각)
        #설치용량 = st.number_input('＊설비용량 [W]', 0, 1000, 400)
        #집광효율 = st.number_input('＊집광효율 (%)', 0.00, 100.00, 21.6)
        #PER= st.number_input('＊Performance Ratio (%)', 0.00, 100.00, 75.00)

    LENGTH = 1043
    WIDTH = 2275
    경사각도 = 'South_45'
    설치용량 = 485
    집광효율 = 20.44
    PER = 75

    with con13 : 
        empty()

    st.caption('                     ', unsafe_allow_html=False)
    st.caption('--------- ', unsafe_allow_html=False)

    A = DF6[제로에너지등급] #제로에너지 취득을 위한 퍼센테이지 정보

    #기준1_에효 1++(비주거용 140 미만)
    x = {'개선전':[hh-139], '개선후':[ii-139]}
    xx = pd.DataFrame(x, index=['에너지효율등급'])

    #기준2_제로에너지 
    y = {'개선전':[A[0]/100*hh], '개선후':[A[0]/100*ii]}
    yy = pd.DataFrame(y, index=['제로에너지'])

    #base와 개선후 표 합치기 = result
    result = pd.concat([xx,yy])
    ## result

    #최대값
    mm = result.max(axis=0)
    mmm = pd.DataFrame(mm, columns=['최대값'])
    mmm = mmm.transpose() 
    ## mmm

    # result와 최대값 표합치기 = result2
    ## st.subheader('단위면적당 필요에너지 비교표')
    result2 = pd.concat([result,mmm])
    ## result2

    st.markdown("#### 3. 신재생 필요 발전 용량 산정")

    con000, con010, con0101, con020, con0201 = st.columns([0.01, 0.7, 0.3, 0.7, 0.3])
    with con010 : 
        st.caption('         ', unsafe_allow_html=False)
        st.markdown("##### ● 필요 에너지 생산량 산정")
        st.caption('         ', unsafe_allow_html=False)

        st.text('① 개선후 철도역사의 단위면적당 연간 1차 에너지 소요량')
        result11 = round(i_room_elex_drop_1차_연.at['개선후', '단위면적당_연간전기사용량'],2)
        r1 = f'{result11}kWh/㎡yr'
        st.markdown(f"###### ▶ :blue[{r1}]")
        #항목1_제로에너지 
        st.text('② (항목1)선택한 ZEB등급 취득을 위해 필요한 단위면적당 연간 1차 에너지 생산량')
        result22 = round(result2.at['제로에너지', '개선후'],2)
        r2 = f'{result22} kWh/㎡yr'
        st.markdown(f"###### ▶ :blue[{r2}]")
        #항목2_에효 1++(비주거용 140 미만)
        st.text('③ (항목2)건축물에너지효율등급 1++등급 취득을 위해 필요한 단위면적당 연간 1차 에너지 생산량')
        result23 = round(result2.at['에너지효율등급', '개선후'],2)
        r3 = f'{result23} kWh/㎡yr'
        st.markdown(f"###### ▶ :blue[{r3}]")
        #결론
        st.text('④ (결론1)단위면적당 연간 필요 1차 에너지 생산량')
        result24 = round(result2.at['최대값', '개선후'],2)
        r4 = f'{result24} kWh/㎡yr'
        st.markdown(f"###### ▶ :blue[{r4}]")
        st.text('⑤ (결론2)단위면적당 연간 필요 2차 에너지 생산량')
        result25 = round(result24/2.75,2)
        r5 = f'{result25} kWh/㎡yr'
        st.markdown(f"###### ▶ :blue[{r5}]")
    
    #설정값으로 인한 산출값
    집광면적 = LENGTH*WIDTH/1000000
    설비용량 = 설치용량/1000

    #지역별 일사량 
    a = DF1[지역]
    ## st.dataframe(a)

    #방위별 경사일사량 = cc
    ## st.subheader('c')
    c = DF2[경사각도]
    ## st.dataframe(c)
    
    #.맑은날 일수  = f
    ## st.subheader('f')
    ## st.dataframe(DF3)
    f = DF3['일수']

    #지역별 수평일사량 = bb
    ## st.subheader('b')
    b= [a[0] / f[0], a[1] / f[1], a[2] / f[2], a[3] / f[3], a[4] / f[4], a[5] / f[5], a[6] / f[6], a[7] / f[7], a[8] / f[8], a[9] / f[9], a[10] / f[10], a[11] / f[11]]
    bb = pd.DataFrame(b, index=['01월', '02월', '03월', '04월', '05월', '06월', '07월', '08월', '09월', '10월', '11월', '12월'], columns=['수평일사량'])
    round(bb['수평일사량'],3)
    ## st.dataframe(bb)

    #경사일사량 = dd
    ## st.subheader('d')
    d = c[0] * b[0], c[0] * b[1], c[0] * b[2], c[0] * b[3], c[0] * b[4], c[0] * b[5], c[0] * b[6], c[0] * b[7], c[0] * b[8], c[0] * b[9], c[0] * b[10], c[0] * b[11]
    dd = pd.DataFrame(d, index=['01월', '02월', '03월', '04월', '05월', '06월', '07월', '08월', '09월', '10월', '11월', '12월'], columns=['경사일사량'])
    ## st.dataframe(dd)

    #일일발전량 = ee
    e = [d[0] * 집광효율 * 집광면적 * PER/1000000, 
    d[1] * 집광효율 * 집광면적 * PER/1000000, 
    d[2] * 집광효율 * 집광면적 * PER/1000000, 
    d[3] * 집광효율 * 집광면적 * PER/1000000, 
    d[4] * 집광효율 * 집광면적 * PER/1000000, 
    d[5] * 집광효율 * 집광면적 * PER/1000000, 
    d[6] * 집광효율 * 집광면적 * PER/1000000, 
    d[7] * 집광효율 * 집광면적 * PER/1000000, 
    d[8] * 집광효율 * 집광면적 * PER/1000000, 
    d[9] * 집광효율 * 집광면적 * PER/1000000, 
    d[10] * 집광효율 * 집광면적 * PER/1000000, 
    d[11] * 집광효율 * 집광면적 * PER/1000000]
    ee = pd.DataFrame(e, index=['01월', '02월', '03월', '04월', '05월', '06월', '07월', '08월', '09월', '10월', '11월', '12월'], columns=['일일발전량'])
    ## st.dataframe(ee)
    
    #월간발전량 = g
    g = [e[0] * f[0], e[1] * f[1], e[2] * f[2], e[3] * f[3], e[4] * f[4], e[5] * f[5], e[6] * f[6], e[7] * f[7], e[8] * f[8], e[9] * f[9], e[10] * f[10], e[11] * f[11]]
    gg = pd.DataFrame(g, index=['01월', '02월', '03월', '04월', '05월', '06월', '07월', '08월', '09월', '10월', '11월', '12월'], columns=['월간발전량'])
    ## st.dataframe(gg)

    #일일발전량_월간발전량 합치기 
    eeeee = pd.concat([ee, gg],axis=1, join='inner')
    #필요 태양광 용량 산정하기
    #모듈 1개당 연간발전량
    B = gg['월간발전량'].sum()

    #모듈 용량 KW로 변환
    D = 설치용량/1000 

    #alt1 필요 태양광 용량 및 면적
    A_alt1 = round(result25*area_air,0) # 전체 건물의 필요한 에너지 생산량 > 공조면적값으로 변경(23.08.02)
    C_alt1 = round(A_alt1/B,0) #필요한 태양광 모듈의 개수
    E_alt1 = round(C_alt1*D,0) #총 필요한 태양광 용량 KW   
    F_alt1 = round(C_alt1*집광면적,0)#총 필요한 집광면적

    ## st.text('■ 선택한 ZEB등급 취득을 위해 필요한 태양광 에너지생산량(단위: kW)')
    ## A_alt1
    
    #표로 만들기
    ## soladata = {'개선후':[A_alt1, C_alt1, F_alt1, E_alt1]}
    ## DF7 = pd.DataFrame(soladata, index=['필요에너지생산량', '필요태양광모듈개수', '필요집광면적', '총태양광용량'])
    ##st.dataframe(DF7)
    
    #지열
    area4 = area2 #지열히트펌프공급면적
    y_alt2_kw = round((area4/0.056051-1609.64)*(air_ratio/100))
        
    # 태양광으로 대체해야할 에너지생산량 계산(전체 건물의 필요한 에너지 생산량-지열히트펌프설치면적)_alt
    # st.markdown("##### ▣ 태양광")

    # 계산
    #필요한 태양광 에너지생산량
    sola_A_alt = A_alt1-y_alt2_kw
    #필요한 태양광 모듈의 개수
    sola_C_alt = round(sola_A_alt/B,0)  
    #총 필요한 태양광 용량 KW
    sola_E_alt = round(sola_C_alt*D, 0)
     #총 필요한 집광면적
    sola_F_alt = round(sola_C_alt*집광면적,0)            

    #홈페이지 나타내기
    with con020 :
        st.markdown("##### ● 태양광 및  지열 필요 용량 산정")
        
        st.text('① 목표등급 달성을 위한 총 필요에너지생산량(공조면적기준)')

        r6 = format(int(A_alt1),',d')
        r7 = f'{r6} kWh/yr'
        
        st.markdown(f"##### ▶ :blue[{r7}]")

        idx1 = [['개선후(태양광)', '개선후(태양광+지열)']]
        columns_list = [('태양광 용량(kW)'), ('태양광 집광면적(㎡)'), ('태양광 모듈개수(EA)'), ('지열 용량(kW)')]
        data1 = np.array([[E_alt1, F_alt1, C_alt1, 0], [sola_E_alt, sola_F_alt, sola_C_alt, y_alt2_kw]])
        col1 = np.array([['태양광', '태양광', '태양광','지열'],['용량(kW)', '집광면적(㎡)', '모듈개수(EA)', '용량(kW)']])
        신재생설치계획 = pd.DataFrame(data1, index = idx1, columns = columns_list)
        st.dataframe(신재생설치계획.style.format("{:,.0f}"))
        
    st.caption('--------', unsafe_allow_html=False)  
    st.markdown("#### 4. 신재생 설치로 인한 기대 개선 효과 분석 ")

    #필요정보 만들기
    #제안1 태양광 전제월간발전량(1차) = g*모듈개수
    ggg = pd.DataFrame(g, columns=['월간발전량'])
    g_all = ggg*C_alt1*2.75
    # g_all

    #제안2 전제월간발전량(1차)
    ggg = pd.DataFrame(g, columns=['태양광_월간발전량'])
    g_all2 = ggg*sola_C_alt*2.75
    g_all2['지열_월간발전량'] = y_alt2_kw / 12*2.75
    g_all2['월간_발전량 합계'] = g_all2['태양광_월간발전량'] + g_all2['지열_월간발전량']
    ## g_all2
    #개선전 월별 에너지 소요령값 출력
    #h
    BASE_연간전체 = h.groupby('Alt').agg(연간전기사용량_전체 = ('kW', 'sum'), 단위면적당_연간전기사용량_전체 = ('kW/m2', 'sum'))
    BASE_월간전체 = h.groupby(['Alt','Month']).agg( 월간전기사용량_전체 = ('kW', 'sum'), 단위면적당_월간전기사용량_전체 = ('kW/m2', 'sum'))
    BASE_연간원별 = h.groupby('index').agg(연간전기사용량_원별 = ('kW', 'sum'), 단위면적당_연간전기사용량_원별 = ('kW/m2', 'sum'))
    BASE_월간원별 = h.groupby(['index','Month']).agg(연간전기사용량_원별 = ('kW', 'sum'), 단위면적당_연간전기사용량_원별 = ('kW/m2', 'sum'))

    BASE_연간전체 = BASE_연간전체.reset_index()
    BASE_월간전체 = BASE_월간전체.reset_index()
    BASE_월간전체.drop(['단위면적당_월간전기사용량_전체'], axis=1, inplace=True)
    BASE_월간전체['개선전_월간소요량']= round(BASE_월간전체['월간전기사용량_전체'] * 2.75,2)
    BASE_월간전체['개선전_CO2발생량'] = round(BASE_월간전체['개선전_월간소요량']*4.781,2)
    BASE_월간전체['개선전_필요소나무'] = round(BASE_월간전체['개선전_월간소요량']*0.1158,2)
    # BASE_월간전체

    #alt 월별 에너지 소요령값 출력
    #i
    ALT_연간전체 = i.groupby('Alt').agg(연간전기사용량_전체 = ('kW', 'sum'), 단위면적당_연간전기사용량_전체 = ('kW/m2', 'sum'))
    ALT_월간전체 = i.groupby(['Alt','Month']).agg( 월간전기사용량_전체 = ('kW', 'sum'), 단위면적당_월간전기사용량_전체 = ('kW/m2', 'sum'))
    ALT_연간원별 = i.groupby('index').agg(연간전기사용량_원별 = ('kW', 'sum'), 단위면적당_연간전기사용량_원별 = ('kW/m2', 'sum'))
    ALT_월간원별 = i.groupby(['index','Month']).agg(연간전기사용량_원별 = ('kW', 'sum'), 단위면적당_연간전기사용량_원별 = ('kW/m2', 'sum'))

    ALT_연간전체 = ALT_연간전체.reset_index()
    ALT_월간전체 = ALT_월간전체.reset_index()
    ALT_월간전체.drop(['단위면적당_월간전기사용량_전체'], axis=1, inplace=True)
    ALT_월간전체['개선후_월간소요량']= round(ALT_월간전체['월간전기사용량_전체'] * 2.75,2)
    ALT_월간전체['개선후(태양광)_신재생발전량'] = g_all['월간발전량']
    ALT_월간전체['개선후(태양광+지열)_신재생발전량'] = g_all2['월간_발전량 합계']
    ALT_월간전체['개선후(태양광)_월간소요량'] = round(ALT_월간전체['개선후_월간소요량'] * (air_ratio/100)-ALT_월간전체['개선후(태양광)_신재생발전량'],2) #공조면적비율 곱
    ALT_월간전체['개선후(태양광+지열)_월간소요량'] = round(ALT_월간전체['개선후_월간소요량'] * (air_ratio/100)-g_all2['월간_발전량 합계'],2) #공조면적비율 곱
    ALT_월간전체['개선후(태양광)_CO2발생량'] = round(ALT_월간전체['개선후(태양광)_월간소요량']*4.781,2)
    ALT_월간전체['개선후(태양광+지열)_CO2발생량'] = round(ALT_월간전체['개선후(태양광+지열)_월간소요량']*4.781,2)
    ALT_월간전체['개선후(태양광)_필요소나무'] = round(ALT_월간전체['개선후(태양광)_CO2발생량']*0.1158,2)
    ALT_월간전체['개선후(태양광+지열)_필요소나무'] = round(ALT_월간전체['개선후(태양광+지열)_CO2발생량']*0.1158,2)
    # ALT_월간전체
    
    # co2발상량 표합침
    co2발생량 = pd.DataFrame(['01월', '02월', '03월', '04월', '05월', '06월', '07월', '08월', '09월', '10월', '11월', '12월'], columns=['month'])
    co2발생량['개선전']=BASE_월간전체['개선전_CO2발생량']
    co2발생량['개선후(태양광)']=ALT_월간전체['개선후(태양광)_CO2발생량']
    co2발생량['개선후(태양광+지열)']=ALT_월간전체['개선후(태양광+지열)_CO2발생량']
    
    # co2발생량
    co2발생량1 = co2발생량.set_index(keys='month', drop=True, append=False, inplace=False, verify_integrity=False)
    
    # 소나무 표합침 
    소나무 = pd.DataFrame(['01월', '02월', '03월', '04월', '05월', '06월', '07월', '08월', '09월', '10월', '11월', '12월'], columns=['month'])
    소나무['개선전']=BASE_월간전체['개선전_필요소나무']
    소나무['개선후(태양광)']=ALT_월간전체['개선후(태양광)_필요소나무']
    소나무['개선후(태양광+지열)']=ALT_월간전체['개선후(태양광+지열)_필요소나무']
    
    소나무1 = 소나무.set_index(keys='month', drop=True, append=False, inplace=False, verify_integrity=False)
    # 소나무1

    #월별에너지소요량 비교표 
    월간소요량비교 = pd.DataFrame(['01월', '02월', '03월', '04월', '05월', '06월', '07월', '08월', '09월', '10월', '11월', '12월'], columns=['month'])
    월간소요량비교['개선전'] = BASE_월간전체['개선전_월간소요량'] * (air_ratio/100) #공조면적비율 곱
    월간소요량비교['개선후(태양광)'] = ALT_월간전체['개선후(태양광)_월간소요량']
    월간소요량비교['개선후(태양광+지열)'] = ALT_월간전체['개선후(태양광+지열)_월간소요량']
    
    #연간소요량 정보 
    BASE_연간소요량_ = round(BASE_월간전체['개선전_월간소요량'].sum(),2) * (air_ratio/100) #공조면적비율 곱
    ALT_월간전체_ = ALT_월간전체['개선후_월간소요량'].sum() * (air_ratio/100) #공조면적비율 곱
    g_all_ = g_all['월간발전량'].sum()
    g_all2_ = g_all2['월간_발전량 합계'].sum()
    ALT_연간소요량_제안1 = round(ALT_월간전체_ - g_all_,2)
    ALT_연간소요량_제안2 = round(ALT_월간전체_ - g_all2_,2)
    row = ['연간에너지소요량(kW)']
    col = ['개선전', '개선후(태양광)', '개선후(태양광+지열)']
    data_ = [[BASE_연간소요량_, ALT_연간소요량_제안1, ALT_연간소요량_제안2]]
    연간발전량비교 =  pd.DataFrame(data = data_, index = row, columns = col)
    ## st.dataframe(연간발전량비교)

# 합계값 정의
    개선전_CO2발생량 = round(co2발생량['개선전'].sum(),2)
    개선후_태양광_CO2발생량 = round(co2발생량['개선후(태양광)'].sum(),2)
    개선후_태양광_지열_CO2발생량 = round(co2발생량['개선후(태양광+지열)'].sum(),2)

    개선전_필요소나무 = round(소나무['개선전'].sum(),2)
    개선후_태양광_필요소나무 = round(소나무['개선후(태양광)'].sum(),2)
    개선후_태양광_지열_필요소나무 = round(소나무['개선후(태양광+지열)'].sum(),2)

    # f'■ 개선전  : 연간에너지소요량 {BASE_연간소요량_}kWh/yr, CO2배출량 {개선전_CO2발생량}kg'
    # f'■ 개선후(태양광) 기대효과 : 연간에너지소요량 {ALT_연간소요량_제안1}kWh/yr, CO2배출량 {개선후_태양광_CO2발생량}kg으로 {개선후_태양광_CO2발생량/개선전_CO2발생량*100}% 절감, {개선전_필요소나무-개선후_태양광_필요소나무:0.0f}개의 소나무를 식재하는 효과'
    # f'■ 개선후(태양광+지열) 기대효과 : 연간에너지소요량 {ALT_연간소요량_제안2}kWh/yr, CO2배출량 {개선후_태양광_지열_CO2발생량}kg으로 {개선후_태양광_지열_CO2발생량/개선전_CO2발생량*100}% 절감, {개선전_필요소나무-개선후_태양광_지열_필요소나무:0.0f}개의 소나무를 식재하는 효과'

    ## st.markdown("##### ②. CO2 배출량 분석")
    # 표만들기
    fig1 = px.bar(co2발생량1, y=['개선전','개선후(태양광)','개선후(태양광+지열)'], title='CO2 발생량 그래프', barmode='group')
    ## fig1

    개선후_제안1_소요량절감률 = round((1-(ALT_연간소요량_제안1/BASE_연간소요량_))*100,2)
    개선후_제안2_소요량절감률 = round((1-(ALT_연간소요량_제안2/BASE_연간소요량_))*100,2)

    개선후_제안1_CO2절감률 = round((1-(개선후_태양광_CO2발생량/개선전_CO2발생량))*100,2)
    개선후_제안2_CO2절감률 = round((1-(개선후_태양광_지열_CO2발생량/개선전_CO2발생량))*100,2)

    개선후_제안1_소나무식재효과 =round(개선후_태양광_필요소나무-개선전_필요소나무,2)
    개선후_제안2_소나무식재효과 =round(개선후_태양광_지열_필요소나무-개선전_필요소나무,2)

    #홈페이지 나타내기 
    con000, con30, con301, con31, con311 = st.columns([0.01, 0.7, 0.3, 0.7, 0.3])

    # 표만들기
    with con30 :
        #연간발전량비교1=연간발전량비교.set_index(['month'])
        #st.markdown("##### (1) 연간에너지 소요량 비교")
        fig3 = px.bar(연간발전량비교, y=['개선전', '개선후(태양광)', '개선후(태양광+지열)'], title="① 연간 에너지 소요량 비교", barmode='group')
        fig3
        
    with con31 :
        월간소요량비교1=월간소요량비교.set_index(['month'])
        #st.markdown("##### (2) 월별에너지 소요량 비교")
        fig2 = px.bar(월간소요량비교1, y=['개선전', '개선후(태양광)', '개선후(태양광+지열)'], title="② 월간 에너지 소요량 비교", barmode='group')
        fig2
    
    st.caption('--------- ', unsafe_allow_html=False)

    # 공사비 
    PR = DF7['가격']
    ACH50가격 = format(PR[0]*int(area_air),',d')
    Chiller_COP가격 = format(PR[1]*int(area_air),',d')
    Fan_total_efficiency가격 = format(PR[2]*int(area_air),',d')
    Occupied가격 = format(PR[3]*int(area_air),',d')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    Pump_efficiency가격 = format(PR[4]*int(area_air),',d')
    heat_recover_effectiveness가격 = format(PR[5]*int(area_air),',d')
    Lighting_power_density가격 = format(PR[6]*int(area_air),',d')
    지열가격1 = 0
    지열가격2 = format(int(PR[7]*y_alt2_kw),',d')
    태양광1 = format(int((PR[8]*C_alt1)),',d')
    태양광2 = format(int((PR[8]*sola_C_alt)),',d')

    st.caption('         ', unsafe_allow_html=False)
    st.markdown("#### 5. 기술 요소별 예상 공사비 산정(공조면적 기준)")
    st.caption('(각 항목 체크시 예상공사비 산정됨)', unsafe_allow_html=False)
    st.caption('         ', unsafe_allow_html=False)
    con000, con001, con002, con003, con004, con005 = st.columns([0.01, 0.3, 0.3, 0.3, 0.3, 0.3])
    st.caption('         ', unsafe_allow_html=False)
    st.caption('         ', unsafe_allow_html=False)
    con000, con006, con007, con008, con009, con010 = st.columns([0.01, 0.3, 0.3, 0.3, 0.3, 0.3])

    with con001 : 
        box2 = st.checkbox('기밀공사')
        if box2 : 
            f'설치비용 : {ACH50가격} 원'
        else : 
            f'설치비용 : 0 원'

    with con002 : 
        box3 = st.checkbox('고효율 냉동기 교체')
        if box3 : 
            f'설치비용 : {Chiller_COP가격} 원'
        else : 
            f'설치비용 : 0 원'

    with con003 : 
        box4 = st.checkbox('고효율 팬 교체')
        if box4 : 
            f'설치비용 : {Fan_total_efficiency가격} 원'
        else : 
            f'설치비용 : 0 원'

    with con004 : 
        box5 = st.checkbox('고효율 이코너마이저 설치')
        if box5 : 
            f'설치비용 : {Occupied가격} 원'
        else : 
            f'설치비용 : 0 원'

    with con005 : 
        box6 = st.checkbox('고효율 펌프 교체')
        if box6 : 
            f'설치비용 : {Pump_efficiency가격} 원'
        else : 
            f'설치비용 : 0 원'

    with con006 : 
        box7 = st.checkbox('고효율 전열교환기  교체')
        if box7 : 
            f'설치비용 : {heat_recover_effectiveness가격} 원'
        else : 
            f'설치비용 : 0 원'

    with con007 : 
        box8 = st.checkbox('고효율 조명 교체')
        if box8 : 
            f'설치비용 : {Lighting_power_density가격} 원'
        else : 
            f'설치비용 : 0 원'

    with con008 : 
        box9 = st.checkbox('신재생(태양광)')
        if box9 : 
            f'설치비용 태양광 : {태양광1} 원'
            f'설치비용 지열 : {지열가격1} 원'
            

        else : 
            f'설치비용 태양광 : 0 원'
            f'설치비용 지열 : 0 원'

    with con009 : 
        box10 = st.checkbox('신재생(태양광+지열)')
        if box10 : 
            f'설치비용 태양광 : {태양광2} 원'
            f'설치비용 지열 : {지열가격2} 원'
        else : 
            f'설치비용 태양광 : 0 원'
            f'설치비용 지열 : 0 원'

    with con010 :
        box11  = st.write(':white_check_mark: 예상 총 공사비')
        if box2 == True :
            box2 = PR[0]*area_air
        else :
            box2 = 0
                
        if box3 == True :
            box3 = PR[1]*area_air
        else :
            box3 = 0

        if box4 == True :
            box4 = PR[2]*area_air
        else :
            box4 = 0

        if box5 == True :
            box5 = PR[3]*area_air
        else :
            box5 = 0

        if box6 == True :
            box6 = PR[4]*area_air
        else :
            box6 = 0

        if box7 == True :
            box7 = PR[5]*area_air
        else :
            box7 = 0

        if box8 == True :
            box8 = PR[6]*area_air
        else :
            box8 = 0

        if box9 == True :
            box9 = (PR[8]*C_alt1)
        else :
            box9 = 0
        
        if box10 == True :
            box10 = (PR[8]*sola_C_alt) + round(PR[7]*y_alt2_kw,0)
        else :
            box10 = 0
        
        개선후_태양광_합계 = format(int(round(box2 + box3 + box4 + box5 + box6 + box7 + box8 + box9,0)),',d')
        개선후_태양광_지열_합계 = format(int(round(box2 + box3 + box4 + box5 + box6 + box7 + box8 + box10,0)),',d')

        tp1 = f'{개선후_태양광_합계} 원'
        tp2 = f'{개선후_태양광_지열_합계} 원'
        st.markdown(f"###### 개선후(태양광) : :blue[{tp1}]")
        st.markdown(f"###### 개선후(태양광+지열) : :blue[{tp2}]")
    
    
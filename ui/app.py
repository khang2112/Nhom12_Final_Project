"""Functional Gradio web app (allowed by Stage 2). Run: python ui/app.py"""
from pathlib import Path
import sys
import os
import sklearn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from importlib import import_module
import numpy as np,pandas as pd,joblib,gradio as gr
import plotly.express as px
from plotly import graph_objects as go
ROOT=Path(__file__).resolve().parents[1]
feature_mod=import_module('03_model_stage2')
from vector_store import nearest_frame
DF=pd.read_csv(ROOT/'data/processed/master_with_clusters.csv')
REG=joblib.load(ROOT/'models/regression_winner.joblib')
CLF=joblib.load(ROOT/'models/classification_winner.joblib')
CLUSTER=joblib.load(ROOT/'models/clustering_pipeline.joblib')
RESULTS=pd.read_json(ROOT/'reports/model_results.json',typ='series')
if RESULTS.get('sklearn_version') != sklearn.__version__:
    raise RuntimeError('Sai phiên bản scikit-learn. Cài đúng requirements.txt hoặc chạy lại pipeline.')
EVAL=pd.read_csv(ROOT/'reports/regression_benchmark.csv')
CLASS_EVAL=pd.read_csv(ROOT/'reports/classification_benchmark.csv')

THEME_CSS='''
.gradio-container {max-width: 1250px !important; margin: auto !important;}
.hero {border-radius: 22px; background: linear-gradient(105deg,#092e39,#075b67 57%,#149393); color:#fff; padding:29px 34px; margin-bottom:14px;}
.hero h1 {color:#fff !important; font-size:32px !important; margin:0 0 10px 0 !important;}
.hero p {color:#d4fbf6 !important; font-size:15px;}
.footer-tip {border-left:4px solid #1fa6a0;padding:8px 15px;background:#eaf7f5;border-radius:7px;color:#124950;}
'''

def predict(continent,year,population,gdp_per_capita,sim_health_index,sim_education_index,sim_trend_index,sim_policy_group):
    values = [year, population, gdp_per_capita, sim_health_index, sim_education_index, sim_trend_index]
    try:
        valid = all(np.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        return 'Vui lòng nhập các giá trị số hữu hạn.', pd.DataFrame()
    if int(year) not in range(1952, 2008, 5) or float(year) != int(year):
        return 'Chọn năm lịch sử từ 1952 đến 2007, bước 5 năm.', pd.DataFrame()
    if continent not in ['Asia','Africa','Americas','Europe','Oceania']:
        return 'Châu lục không hợp lệ.', pd.DataFrame()
    if sim_policy_group not in ['Higher simulated investment','Moderate simulated investment','Lower simulated investment','Unknown']:
        return 'Nhóm mô phỏng không hợp lệ.', pd.DataFrame()
    if any(not 0 <= float(v) <= 100 for v in values[3:]):
        return 'Chỉ số mô phỏng phải trong khoảng 0–100.', pd.DataFrame()
    if population<=0 or gdp_per_capita<=0:
        return 'Vui lòng nhập dân số và GDP/người lớn hơn 0.',pd.DataFrame()
    row=pd.DataFrame([dict(continent=continent,year=int(year),population=float(population),gdp_per_capita=float(gdp_per_capita),
       sim_health_index=float(sim_health_index),sim_education_index=float(sim_education_index),
       sim_trend_index=float(sim_trend_index),sim_policy_group=sim_policy_group)])
    X=feature_mod.features(row)
    expected=float(REG.predict(X)[0]);chance=float(CLF.predict_proba(X)[0,1]);classification=int(CLF.predict(X)[0])
    xp=CLUSTER['preprocessor'].transform(X)
    xc=CLUSTER['pca'].transform(xp) if CLUSTER['representation']=='PCA features' else xp
    segment=int(CLUSTER['model'].predict(xc)[0])
    label='≥ 70 năm' if classification==1 else '< 70 năm'
    md=f'''### Kết quả mô hình
**Tuổi thọ dự đoán:** {expected:.1f} năm
**Phân loại dự đoán (mô hình độc lập):** {label}
**Xác suất lớp tuổi thọ ≥70:** {chance:.1%}
**Nhóm K-Means:** Cụm {segment}

> Demo lịch sử 1952–2007; sim_* là mô phỏng. Hồi quy và phân loại học độc lập. Không dùng làm quyết định chính sách.'''
    vector = CLUSTER['pca'].transform(xp)[0]
    try:
        nearest = nearest_frame(vector)
        similar = nearest[['record_id', 'distance_l2']].merge(
            DF[['record_id','country','year','continent','life_expectancy','gdp_per_capita']],
            on='record_id', how='left', validate='one_to_one').drop(columns='record_id')
    except Exception:
        md += '\n\n**Vector DB chưa sẵn sàng.** Kiểm tra dịch vụ ChromaDB; 3 kết quả mô hình vẫn hiển thị.'
        similar = pd.DataFrame()
    if (expected >= 70) != bool(classification):
        md += '\n\n**Lưu ý:** hồi quy và phân loại đang bất đồng ở ngưỡng 70.'
    similar=similar.rename(columns={'country':'Quốc gia','year':'Năm','continent':'Châu lục','life_expectancy':'Tuổi thọ thực','gdp_per_capita':'GDP/người thực'})
    return md,similar


def scatter():
    f=px.scatter(DF,x='gdp_per_capita',y='life_expectancy',color='continent',hover_name='country',
      hover_data=['year','Cluster_Label'],log_x=True,opacity=.53,
      labels={'gdp_per_capita':'GDP/người (log)','life_expectancy':'Tuổi thọ (năm)','continent':'Châu lục'},
      title='Mối quan hệ GDP/người và tuổi thọ')
    f.update_layout(height=440,margin=dict(l=20,r=20,t=55,b=10),template='plotly_white')
    return f

def trend():
    trends=DF.groupby(['year','continent'],as_index=False).life_expectancy.mean()
    f=px.line(trends,x='year',y='life_expectancy',color='continent',markers=True,
        labels={'year':'Năm','life_expectancy':'Tuổi thọ TB (năm)','continent':'Châu lục'},title='Xu hướng tuổi thọ theo châu lục')
    f.update_layout(height=410,margin=dict(l=20,r=20,t=55,b=10),template='plotly_white')
    return f

with gr.Blocks(title='Atlas Insight | Data Analyst AI') as demo:
    gr.HTML('''<div class="hero"><div style="font-size:12px;letter-spacing:2px;font-weight:700;color:#b7f4e9">DATA ANALYST AI · FINAL TEAM PROJECT</div><h1>ATLAS INSIGHT <span style="font-weight:300">/ AI Dashboard</span></h1><p>Khám phá 1.704 quan sát · 142 quốc gia · PCA / K-Means / Regression / Classification</p></div>''')
    with gr.Row():
        gr.Markdown('**Dữ liệu thực:** Gapminder 1952–2007  |  **Dữ liệu bổ sung:** mô phỏng có kiểm soát  |  **Chia tập:** kiểm tra bằng các quốc gia chưa thấy')
    with gr.Tabs():
      with gr.Tab('🎯 Dự đoán AI'):
        gr.Markdown('### Nhập thông tin để dự đoán tuổi thọ và phân nhóm quốc gia-năm')
        with gr.Row():
          with gr.Column(scale=3):
            with gr.Row():
                continent=gr.Dropdown(choices=['Asia','Africa','Americas','Europe','Oceania'],label='Châu lục',value='Asia')
                year=gr.Slider(1952,2007,step=5,value=2002,label='Năm lịch sử')
            with gr.Row():
                population=gr.Number(label='Dân số',value=25000000,minimum=100)
                gdp=gr.Number(label='GDP bình quân/người (USD)',value=5000,minimum=1)
            gr.Markdown('**Chỉ số tổng hợp giả lập** – dùng để trình diễn phép nối dữ liệu động')
            with gr.Row():
                health=gr.Slider(0,100,value=70,label='sim_health_index')
                edu=gr.Slider(0,100,value=65,label='sim_education_index')
            with gr.Row():
                tr=gr.Slider(0,100,value=70,label='sim_trend_index')
                pol=gr.Dropdown(['Higher simulated investment','Moderate simulated investment','Lower simulated investment','Unknown'],value='Moderate simulated investment',label='sim_policy_group')
            button=gr.Button('✨ Phân tích bằng AI',variant='primary',size='lg')
          with gr.Column(scale=2):
            result=gr.Markdown('### Kết quả sẽ hiển thị tại đây\n\nĐiền thông tin bên trái và nhấn **Phân tích bằng AI**.')
            rows=gr.Dataframe(label='5 quan sát gần nhất từ ChromaDB (PCA / L2)',interactive=False)
        button.click(predict,[continent,year,population,gdp,health,edu,tr,pol],[result,rows])
      with gr.Tab('📈 Trực quan hóa'):
        gr.Plot(value=scatter(),label='Mối quan hệ đa biến')
        gr.Plot(value=trend(),label='Xu hướng thời gian')
        with gr.Row():
            gr.Image(value=str(ROOT/'charts/11_pca_cumulative_variance.png'),label='PCA cumulative variance',interactive=False)
            gr.Image(value=str(ROOT/'charts/13_kmeans_2d_clusters.png'),label='K-Means segmentation',interactive=False)
      with gr.Tab('🧪 Kết quả đánh giá'):
        gr.Markdown(f'''### Benchmark mô hình thực tế\n**PCA:** {int(RESULTS['pca_components'])} thành phần, giữ lại {float(RESULTS['explained_variance']):.1%} phương sai.
**K-Means:** K={int(RESULTS['chosen_k'])}, Silhouette={float(RESULTS['winning_silhouette']):.3f}.
**Mô hình hồi quy tốt nhất:** {RESULTS['winning_regression']}
**Mô hình phân loại tốt nhất:** {RESULTS['winning_classification']}''')
        with gr.Row():
            gr.Dataframe(value=EVAL,label='MSE · RMSE · R²',interactive=False)
            gr.Dataframe(value=CLASS_EVAL,label='Accuracy · Precision · Recall · F1',interactive=False)
        with gr.Row():
            gr.Image(value=str(ROOT/'charts/14_regression_actual_vs_predicted.png'),label='Actual vs Predicted',interactive=False)
            gr.Image(value=str(ROOT/'charts/16_confusion_matrix.png'),label='Confusion Matrix',interactive=False)
      with gr.Tab('📚 Nguồn dữ liệu & Giới hạn'):
        gr.Markdown('''### Truy xuất dữ liệu và phạm vi hợp lệ
- **Nguồn dữ liệu thực:** [Gapminder Dataset – Plotly mirror](https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv)
- **Giai đoạn:** 1952–2007, không phản ánh bối cảnh năm 2026.
- **Hai feed theo địa lý:** cả hai trích từ cùng nguồn gốc (không phải hai tổ chức độc lập).
- **Mô phỏng:** `sim_health_index`, `sim_education_index`, `sim_trend_index`, `sim_policy_group`; không được hiểu là số đo thực.
- **Phân nhóm:** minh họa phân cụm quan sát quốc gia-năm, không phải phân khúc khách hàng.
- **Phân loại:** ngưỡng tuổi thọ **70 năm** do nhóm lựa chọn; không phải tiêu chuẩn y tế chính thức.
- **Chỉ số:** chia train/test theo **quốc gia**, không để cùng một quốc gia trong cả hai tập.
- **Phạm vi sử dụng:** chỉ dành cho bài thực hành AI; không dùng để dự báo chính sách hoặc đời sống thực.''')

if __name__=='__main__':
    demo.launch(server_name=os.getenv('GRADIO_SERVER_NAME','127.0.0.1'),server_port=7860,show_error=True,share=False,inbrowser=False,css=THEME_CSS,theme=gr.themes.Soft(primary_hue='teal',neutral_hue='slate'))

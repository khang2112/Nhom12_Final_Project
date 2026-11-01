"""Build concise PDF report/slides and a read-only notebook from current artifacts."""
from pathlib import Path
import json
from html import escape
import pandas as pd
import matplotlib
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, PageBreak, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

ROOT=Path(__file__).resolve().parents[1]
R=json.loads((ROOT/'reports/model_results.json').read_text())
C=json.loads((ROOT/'reports/crawl_audit.json').read_text())
REG=pd.read_csv(ROOT/'reports/regression_benchmark.csv')
CLF=pd.read_csv(ROOT/'reports/classification_benchmark.csv')
TEAL=colors.HexColor('#087C80'); INK=colors.HexColor('#173942')
fonts=Path(matplotlib.get_data_path())/'fonts/ttf'
for n,f in [('DV','DejaVuSans.ttf'),('DVB','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(n,str(fonts/f)))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DVB')
styles={n:ParagraphStyle(n,fontName='DVB' if n in ('title','head') else 'DV',fontSize=z,leading=z*1.45,textColor=INK,spaceAfter=9) for n,z in [('title',22),('head',13),('body',11),('small',9)]}


def para(s,style='body'):return Paragraph(s,styles[style])
def table(headers,rows,width=495):
    t=Table([[para(str(x),'small') for x in row] for row in [headers]+rows],colWidths=[width/len(headers)]*len(headers),repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8F3F3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#C5D9DD')),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    return t

def chart(n,maxheight=245,width=480):
    path=next((ROOT/'charts').glob(f'{n:02d}_*.png'));w,h=PILImage.open(path).size;scale=min(width/w,maxheight/h)
    return Image(str(path),width=w*scale,height=h*scale)

def footer(c,d):
    c.saveState();c.setFont('DV',8);c.setFillColor(TEAL);c.drawString(45,24,'NHÓM 12 / ATLAS INSIGHT / BẢN ĐÃ SỬA');c.drawRightString(d.pagesize[0]-45,24,str(d.page));c.restoreState()


def report():
    story=[]
    def page(title):
        if story:story.append(PageBreak())
        story.append(para(title,'title'))
    def p(s):story.append(para(s))
    def t(h,r):story.append(table(h,r));story.append(Spacer(1,12))
    page('1. Tổng quan và kết quả')
    p('Atlas Insight phân tích quốc gia theo năm và trả 3 kết quả: tuổi thọ dự đoán, nhãn tuổi thọ ≥70 năm và cụm hồ sơ. Một dòng là một quốc gia tại một năm.')
    t(['Dữ liệu','Mô hình'],[['1.704 dòng AI, 142 nước, 1952–2007',f"PCA {R['pca_components']} PC, giữ {R['explained_variance']:.2%} phương sai"],['3.104 dòng đa nguồn; 1.704 khóa nước–năm',f"K={R['chosen_k']}; Silhouette={R['winning_silhouette']:.4f}"],['Scrapy tải 12.768 dòng, 8 thuộc tính',f"Hồi quy: {R['winning_regression']}, RMSE {R['best_regression']['RMSE']:.3f} năm"],['375 ô mô phỏng thiếu ở nhánh AI',f"Phân loại: {R['winning_classification']}, F1 macro {R['best_classification']['F1_macro']:.3f}"]])
    p('<b>Đã sửa:</b> crawler thật, missingno, xử lý ngoại lai, imputation trong train, validation riêng, model cùng phiên bản, UI truy vấn vector thật và Docker đa container.')
    p('<b>Đã kiểm chứng:</b> 8 test, UI/Chroma HTTP và truy vấn sau restart DB. Docker Engine chưa có trong môi trường sửa; chạy Compose hoặc chờ GitHub Actions xanh trước khi xác nhận Docker thành công.')
    t(['Người','Báo cáo khoảng 4–5 phút'],[['1','Nguồn, Scrapy, concat và merge'],['2','Lưu trữ, làm sạch, insight EDA'],['3','Feature, chia tập, 6 thuật toán và metrics'],['4','UI, Docker, demo và kết luận']])
    p('Giữ 5 tác giả của báo cáo gốc; nhóm tự chọn 4 người đứng báo cáo. Không cần đọc lại toàn bộ code vẽ giao diện.')
    page('2. Thu thập và tích hợp dữ liệu')
    p('<b>00_crawl_worldbank.py:</b> Scrapy gọi API World Bank SP.DYN.LE00.IN, phân trang, retry và tôn trọng robots.txt. Chỉ ghi dữ liệu sau khi tải đủ trang và đạt ≥300 dòng. Có timestamp, URL và SHA256 trong crawl_audit.json.')
    t(['Thao tác','Kết quả / lý do'],[['01: chuẩn hóa Gapminder','1.704 dòng; sửa ISO Triều Tiên thành PRK; tạo record_id'],['09: lọc World Bank','1.400 dòng tương ứng nước/năm trong Gapminder'],['10: inner join','1.394 cặp đủ tuổi thọ; chênh lệch TB 1,795 năm'],['11: concat','1.704 + 1.400 = 3.104 dòng; giữ data_source'],['12: left merge, many_to_one','Bổ sung GDP/dân số và sim_*; giữ tất cả dòng nguồn'],['13–14: điền và audit','6 tuổi thọ từ Gapminder có flag; 690 ô sim_* được điền']])
    p('<b>Khóa:</b> nhánh AI dùng record_id duy nhất; nhánh nguồn dùng (data_source, iso_alpha, year). Không đưa thẳng 3.104 dòng vào ML vì trùng đơn vị nước–năm.')
    p('<b>sim_*:</b> sinh từ GDP, năm và nhiễu; chỉ dùng thực hành. GDP/dân số trên dòng World Bank vẫn lấy từ Gapminder, có economic_data_source để truy vết.')
    p('<b>Khi cô hỏi:</b> concat nối hàng; merge thêm thông tin bằng khóa. Inner dùng cho đối chiếu cặp chung; left dùng khi phải bảo toàn bảng nguồn.')
    page('3. Làm sạch và EDA')
    p('02 xử lý dữ liệu để mô tả: median health/education theo châu lục, ffill/bfill trend trong từng nước, Unknown cho policy. Không xóa hàng. Missingno hiển thị cấu trúc thiếu; so sánh mean, median, std trước/sau.')
    story.append(chart(17,220))
    p('<b>ML dùng quy trình riêng:</b> 03 đọc master_sparse; FeatureCleaner chỉ học median và giới hạn IQR từ train. Transform test chỉ dùng thống kê đã học, không bfill theo các dòng test.')
    t(['Phát hiện','Ảnh hưởng đến ML'],[['GDP/dân số có đuôi dài','log1p rồi giới hạn IQR cho 2 feature này; không cắt target tuổi thọ'],['Spearman GDP–tuổi thọ ≈0,8265','Giữ GDP làm feature; tương quan không chứng minh nhân quả'],['Lớp/vùng không cân bằng','Chia theo nước, stratify châu lục ở outer split; class_weight balanced'],['Một nước có 12 năm','Giữ nước tách biệt giữa fit, validation và test']])
    page('4. Mô hình và đánh giá')
    p('X gồm 6 cột số và 2 danh mục. log_population/log_gdp giảm lệch; StandardScaler chuẩn hóa số; OneHotEncoder mã hóa continent/policy. Không đưa tuổi thọ, target hay Cluster_Label vào X.')
    p('<b>Supervised:</b> 90 nước fit, 23 validation, 29 test. Chọn winner theo RMSE/F1 macro trên validation; refit 113 nước rồi đánh giá test 348 dòng. PCA chỉ dùng cho clustering.')
    t(['Thuật toán','MSE','RMSE','R²'],[[r.algorithm,f'{r.MSE:.3f}',f'{r.RMSE:.3f}',f'{r.R2:.3f}'] for r in REG.itertuples()])
    t(['Thuật toán','Accuracy','F1 macro'],[[r.algorithm,f'{r.Accuracy:.2%}',f'{r.F1_macro:.3f}'] for r in CLF.itertuples()])
    p(f"<b>PCA/K-Means:</b> {R['pca_components']} PC giữ {R['explained_variance']:.2%}; thử K=2..8 trên dữ liệu gốc và PCA. Chọn Silhouette cao nhất: K={R['chosen_k']}, {R['winning_silhouette']:.4f}. Ánh xạ nhãn về CSV/SQLite.")
    story.append(chart(12,170))
    p('RMSE nhỏ tốt hơn; R² không phải Accuracy. F1 macro cân bằng đóng góp 2 lớp. Silhouette còn thấp nên cụm chỉ dùng khám phá.')
    page('5. Database và giao diện')
    t(['Thành phần','Code / luồng'],[['SQLite','02 ghi/đọc master_sparse, master_clean; 03 ghi master_with_clusters'],['ChromaDB','04 dùng model PCA đã lưu, upsert vector 7 chiều; đọc lại DataFrame và so sánh vector'],['UI','Nạp 3 joblib; validate input; features → predict; trả tuổi thọ, nhãn/xác suất và cụm'],['Tìm tương đồng','UI gọi ChromaDB query thật; 5 hàng gần nhất kèm distance_l2'],['Docker','web và chroma tách container; Web gọi chroma:8000, dữ liệu vector lưu volume']])
    p('start_app.py chờ DB sẵn sàng, đồng bộ vector rồi mở Gradio. UI không huấn luyện lại khi bấm nút. Không có LLM trong luồng dự đoán.')
    p('Input không hợp lệ như NaN, dân số/GDP ≤0, năm ngoài phạm vi hoặc chỉ số ngoài 0–100 được từ chối. Nếu 2 model bất đồng ở ngưỡng 70, UI hiển thị lưu ý; không ép kết quả cho giống nhau.')
    p('<b>Chạy:</b> docker compose up --build -d --wait<br/><b>Mở:</b> http://localhost:7860<br/><b>Test:</b> docker compose exec web python -m pytest tests -q')
    p('GitHub Actions chạy Compose, test HTTP, restart database và chụp 4 tab UI. Ảnh Docker nằm trong artifact docker-ui-evidence sau khi workflow chạy thành công; không dùng ảnh cũ để chứng minh bản mới.')
    page('6. Các file cần mở khi cô hỏi')
    t(['File','Vai trò'],[['00_crawl_worldbank.py','Thu thập World Bank thật bằng Scrapy'],['01_prepare_data.py','Gapminder, mô phỏng, khóa và ghép AI'],['02_storage_clean_eda.py','SQLite, thiếu, thống kê và 11 hình EDA'],['03_model_stage2.py','6 thuật toán, validation/test, metrics và lưu model'],['transforms.py','Imputation và IQR clipping học trong train'],['04_chromadb_vector.py','Nạp vector và kiểm tra đọc ngược'],['vector_store.py','Client local/HTTP, DataFrame, nearest query'],['09–14','Chuẩn hóa, đối chiếu, concat/merge và audit đa nguồn'],['ui/app.py; start_app.py','Giao diện, inference; khởi động an toàn'],['tests/; .github/workflows/verify.yml','Test dữ liệu, mô hình, live DB, UI và Docker CI'],['run_pipeline.py; 07; 08','Tái tạo đầu ra; tạo PDF/notebook; xuất notebook copy'],['05_bonus_web_crawler.py; 06_bonus_ollama_text.py','Tham khảo ngoài luồng chính, không dùng làm bằng chứng rubric']])
    p('notebook dùng output hiện tại và chỉ đọc artifact khi Run All; không ghi đè model. reports/ lưu bảng kiểm toán; charts/ lưu biểu đồ; models/ là artifact nhị phân, không phải source code.')
    page('7. Phân công và kịch bản demo')
    t(['Người','Mở file / lời chốt'],[['1','00, 11, 12: “3.104 là dòng theo nguồn; chỉ 1.704 country-year duy nhất dùng ML.”'],['2','02, 04, charts 06/07/17: “Thiếu được xử lý theo loại biến; imputation ML chỉ học train.”'],['3','03, validation_benchmark, regression/classification_benchmark: “Chọn winner trên validation, báo cáo test riêng.”'],['4','UI, Compose, smoke_http: “3 mô hình độc lập; các dòng tương tự lấy thật từ vector DB.”']])
    p('<b>Demo 3 phút:</b><br/>1. Mở UI, giữ mặc định Asia / 2002 / dân số 25 triệu / GDP 5.000 / sim 70–65–70.<br/>2. Bấm Phân tích bằng AI; đọc 3 kết quả và bảng khoảng cách vector.<br/>3. Mở Trực quan hóa, chỉ quan hệ GDP–tuổi thọ.<br/>4. Mở Kết quả đánh giá, giải thích vì sao chọn winner.<br/>5. Chỉ test/CI và kết thúc bằng giới hạn dữ liệu lịch sử.')
    p('<b>Khuyến nghị:</b><br/>• Dùng cụm để chọn hồ sơ cần nghiên cứu sâu, kiểm chứng bằng nguồn cập nhật.<br/>• Giữ mô hình đơn giản làm baseline; ưu tiên dữ liệu thật và sai số theo vùng.<br/>• Cảnh báo khi các mô hình bất đồng, để người dùng xem lại thay vì tự ra quyết định.')
    p('Không đọc toàn bộ báo cáo. Mỗi người giải thích input → xử lý → output và một lý do kỹ thuật trong phần mình.')
    page('8. Câu hỏi ngắn và checklist')
    t(['Câu hỏi','Trả lời'],[['Tại sao PCA 7 mà hình 2D?','7 chiều để phân cụm; 2D chỉ là phép chiếu để vẽ.'],['K=2 có phải target có 2 lớp?','Không. K chọn qua Silhouette, clustering không dùng target.'],['R² 0,778 có phải đúng 77,8%?','Không. Đây là mức giải thích biến thiên theo sai số bình phương.'],['Tại sao F1 macro?','Lớp mất cân bằng; macro cho mỗi lớp trọng số ngang nhau.'],['Tại sao 2 model bất đồng?','Hồi quy và classifier học độc lập; UI đã báo bất đồng.'],['3.104 dòng có dùng train?','Không; bảng này phục vụ tích hợp và đối chiếu nguồn.'],['Đã chạy Docker chưa?','Đã viết cấu hình/CI; chỉ xác nhận chạy khi Compose hoặc workflow thành công.']])
    p('<b>Trước khi báo cáo:</b> cài đúng requirements; chạy 8 test; mở UI; xác nhận CI Docker; tải ảnh docker-ui-evidence; thống nhất số liệu trong reports; tập nói 4 người.')
    p('<b>Giới hạn:</b> dữ liệu lịch sử 1952–2007; sim_* giả lập; nhãn 70 do nhóm chọn; chưa chứng minh nhân quả, hiệu chuẩn xác suất hay dự báo tương lai. Silhouette khoảng 0,288 cho thấy cụm tách chưa mạnh.')
    p('Nguồn: Grading Rubric.xlsx (6 mục, tổng 10 điểm); code và reports hiện tại; Gapminder/Plotly; World Bank SP.DYN.LE00.IN. Crawl được ghi ngày '+C['downloaded_at_utc'][:10]+'.')
    path=ROOT/'docs/Bao_Cao_Final.pdf';path.parent.mkdir(exist_ok=True)
    SimpleDocTemplate(str(path),pagesize=(595.28,841.89),leftMargin=50,rightMargin=50,topMargin=40,bottomMargin=43,title='Nhóm 12 báo cáo ngắn bản đã sửa').build(story,onFirstPage=footer,onLaterPages=footer)


def slides():
    slides=[
        ('Atlas Insight', ['Phân tích quốc gia–năm bằng Python và AI','3 đầu ra: tuổi thọ, nhãn ≥70 và cụm hồ sơ','Nhóm 12 • 4 người trình bày'],None),
        ('Dữ liệu và tích hợp', ['Scrapy World Bank: 12.768 dòng, 8 trường','Gapminder 1.704 + World Bank 1.400 = 3.104 dòng theo nguồn','ML dùng 1.704 country-year duy nhất; sim_* là mô phỏng'],None),
        ('Làm sạch và EDA', ['Median theo nhóm, Unknown; missingno và phân phối trước/sau','ML: imputation/IQR fit train, log GDP/dân số','Spearman GDP–tuổi thọ ≈0,8265; không suy luận nhân quả'],6),
        ('PCA và phân cụm', [f"{R['pca_components']} PC giữ {R['explained_variance']:.2%} phương sai",f"Thử K=2..8; chọn K={R['chosen_k']}, Silhouette {R['winning_silhouette']:.4f}",'Nhãn cụm được ghi lại vào CSV và SQLite'],12),
        ('Hồi quy và phân loại', [f"{R['winning_regression']}: RMSE {R['best_regression']['RMSE']:.3f} năm; R² {R['best_regression']['R2']:.3f}",f"{R['winning_classification']}: Accuracy {R['best_classification']['Accuracy']:.2%}; F1 macro {R['best_classification']['F1_macro']:.3f}",'90 nước fit / 23 validation / 29 test; chọn winner bằng validation'],16),
        ('Lưu trữ và triển khai', ['SQLite: bảng dữ liệu; Chroma: vector PCA, round-trip DataFrame','UI gọi 3 model và truy vấn 5 vector gần nhất thật','Docker: Web + Chroma, volume; CI kiểm tra và chụp UI'],None),
        ('Demo và khuyến nghị', ['Nhập hồ sơ → dự đoán → xem vector neighbors → xem benchmark','Dùng cụm để nghiên cứu sâu; kiểm chứng nguồn cập nhật','Giữ baseline đơn giản; cảnh báo bất đồng giữa model'],None),
        ('Kết quả kiểm tra và giới hạn', ['8 test PASS; HTTP UI/DB PASS; DB restart vẫn truy vấn được','Docker Engine chưa chạy trong môi trường sửa; kiểm tra workflow trước báo cáo','1952–2007; sim_* giả lập; không dùng quyết định chính sách'],None),
    ]
    story=[]
    for i,(title,lines,n) in enumerate(slides):
        if i:story.append(PageBreak())
        title_style=ParagraphStyle('slide_title',fontName='DVB',fontSize=32,leading=42,textColor=TEAL,spaceAfter=20)
        body_style=ParagraphStyle('slide_body',fontName='DV',fontSize=20 if n else 26,leading=29 if n else 39,textColor=INK,spaceAfter=12 if n else 22)
        story.append(Paragraph(title,title_style))
        for line in lines:story.append(Paragraph(line,body_style))
        if n:story.append(chart(n,maxheight=205,width=730))
    path=ROOT/'slide/Slides_Final_Project.pdf';path.parent.mkdir(exist_ok=True)
    SimpleDocTemplate(str(path),pagesize=(960,540),leftMargin=50,rightMargin=50,topMargin=35,bottomMargin=40,title='Nhóm 12 trình bày ngắn').build(story,onFirstPage=footer,onLaterPages=footer)


def notebook():
    cells=[]
    def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
    def code(s):cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s.splitlines(True)})
    md('# Atlas Insight — Nhóm 12\nNotebook đọc kết quả đã tạo; không ghi đè data/model. Chạy `python run_pipeline.py` khi cần tái tạo.\nNguồn: https://api.worldbank.org/v2/country/all/indicator/SP.DYN.LE00.IN và https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')
    code("from pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nif not (ROOT/'reports').exists(): ROOT = ROOT.parent\nassert (ROOT/'reports').exists()\nR=json.loads((ROOT/'reports/model_results.json').read_text())")
    for title,expr in [('Crawl Scrapy World Bank',"json.loads((ROOT/'reports/crawl_audit.json').read_text())"),('Tích hợp hai nguồn',"pd.read_csv(ROOT/'reports/multisource_audit.csv')"),('Missingness trước/sau',"pd.read_csv(ROOT/'reports/missingness_before.csv')"),('Thống kê đơn biến',"pd.read_csv(ROOT/'reports/univariate_statistics.csv')"),('Bảng liên ngẫu',"pd.read_csv(ROOT/'reports/crosstab_continent_policy.csv')")]:
        md('## '+title);code('display('+expr+')')
    insights=[
        '375 ô thiếu ở sim_*. Dữ liệu quan trắc Gapminder không bị cố ý xóa.',
        'Median thay đổi hình dạng phân phối; so sánh thêm mean/median/std trong imputation_impact.csv.',
        'Tuổi thọ phân tán theo nước/năm; giữ year và continent trong feature.',
        'IQR tuổi thọ không đánh dấu ngoại lai. GDP/dân số được log và IQR clipping trong pipeline ML; không cắt target.',
        'Phân bố châu lục lệch. Outer split phân tầng châu lục theo danh sách nước.',
        'Spearman GDP–tuổi thọ ≈0,8265; gợi ý feature dự báo, không phải nhân quả.',
        'GDP có đuôi dài; log1p và đối chứng tuyến tính/cây kiểm tra tính phi tuyến.',
        'Khác biệt châu lục hỗ trợ OneHotEncoder(continent).',
        'Tỷ lệ lớp không đều: class_weight balanced và F1 macro.',
        'Một nước có nhiều năm: tách nước giữa fit/validation/test, không diễn giải thành dự báo tương lai.',
        f"PCA giữ {R['explained_variance']:.2%} với {R['pca_components']} PC; không phải Accuracy.",
        f"K={R['chosen_k']} chọn theo Silhouette {R['winning_silhouette']:.4f}; WCSS dùng tham khảo Elbow.",
        'Hình 2D chỉ là phép chiếu; K-Means dùng các PC được chọn, không dùng target.',
        f"Winner được chọn bằng validation; test RMSE {R['best_regression']['RMSE']:.3f} năm.",
        'Residual = thật − dự đoán. Cần xem cấu trúc lỗi, không chỉ một metric.',
        f"F1 macro {R['best_classification']['F1_macro']:.3f}; phân biệt lỗi từng lớp qua confusion matrix.",
        'Missingno cho thấy cấu trúc thiếu trong các chỉ số mô phỏng, hỗ trợ kiểm tra trước điền.'
    ]
    for n,insight in enumerate(insights,1):
        f=next((ROOT/'charts').glob(f'{n:02d}_*.png'))
        md('## '+f.stem.replace('_',' '));code(f"display(Image(filename=str(ROOT/'charts/{f.name}'), width=800))");md('**So what?** '+insight)
    for title,file in [('Validation và chọn winner','validation_benchmark.csv'),('Regression test','regression_benchmark.csv'),('Classification test','classification_benchmark.csv'),('ChromaDB đọc ngược DataFrame','chroma_roundtrip.csv')]:
        md('## '+title);code(f"display(pd.read_csv(ROOT/'reports/{file}').head(10))")
    md('## Kiểm tra triển khai\nHTTP UI/Chroma và restart đã test. Docker cần workflow xanh hoặc chạy Compose trên máy nhóm. Xem README để demo.');code("display(json.loads((ROOT/'reports/integration_status.json').read_text()))")
    obj={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':5}
    import uuid
    for c in cells:c['id']=uuid.uuid4().hex[:8]
    (ROOT/'notebooks/Project_Stage1_Stage2_NHOM12.ipynb').write_text(json.dumps(obj,ensure_ascii=False,indent=1),encoding='utf-8')


if __name__=='__main__':
    report();slides();notebook()
    print('Built concise report PDF, slide PDF and read-only notebook.')

"""Capture real UI screenshots after Docker startup; requires playwright."""
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
out = root/'screenshots/current'; out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width':1440, 'height':1000}, device_scale_factor=1)
    page.goto(os.getenv('APP_URL','http://127.0.0.1:7860'), wait_until='networkidle')
    page.get_by_role('button', name='✨ Phân tích bằng AI').click()
    page.get_by_text('Tuổi thọ dự đoán:', exact=False).wait_for(timeout=60000)
    page.screenshot(path=str(out/'01_prediction.png'), full_page=True)
    for i, name in enumerate(['📈 Trực quan hóa','🧪 Kết quả đánh giá','📚 Nguồn dữ liệu & Giới hạn'], 2):
        page.get_by_role('tab', name=name).click()
        page.screenshot(path=str(out/f'{i:02d}_tab.png'), full_page=True)
    browser.close()

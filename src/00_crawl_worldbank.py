# SOURCE: https://api.worldbank.org/v2/country/all/indicator/SP.DYN.LE00.IN
# API documentation: https://datahelpdesk.worldbank.org/knowledgebase/topics/125589
"""Scrapy crawl of the official public World Bank API, with pagination and audit."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import scrapy
from scrapy.crawler import CrawlerProcess

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://api.worldbank.org/v2/country/all/indicator/SP.DYN.LE00.IN?format=json&date=1952:2007&per_page=20000'


class WorldBankSpider(scrapy.Spider):
    name = 'worldbank_life_expectancy'
    allowed_domains = ['api.worldbank.org']
    start_urls = [URL]
    custom_settings = {'ROBOTSTXT_OBEY': True, 'DOWNLOAD_DELAY': 1,
                       'CONCURRENT_REQUESTS_PER_DOMAIN': 1, 'RETRY_TIMES': 3,
                       'DOWNLOAD_TIMEOUT': 60,
                       'USER_AGENT': 'Nhom12-Academic-DataAnalysis/1.0',
                       'LOG_LEVEL': 'INFO'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self.pages = set()
        self.meta = None

    def parse(self, response):
        payload = response.json()
        if not isinstance(payload, list) or len(payload) != 2:
            raise ValueError('Unexpected World Bank response')
        meta, records = payload
        self.meta = meta
        self.pages.add(int(meta['page']))
        self.records.extend(records)
        if int(meta['page']) == 1:
            for page in range(2, int(meta['pages'])+1):
                yield scrapy.Request(URL + f'&page={page}', callback=self.parse)


def main():
    process = CrawlerProcess()
    crawler = process.create_crawler(WorldBankSpider)
    process.crawl(crawler)
    process.start()
    spider = crawler.spider
    if not spider or not spider.meta or len(spider.pages) != int(spider.meta['pages']):
        raise RuntimeError('Crawl incomplete; existing source was not overwritten')
    records = spider.records
    rows = [{'iso_alpha': r['countryiso3code'], 'country_worldbank': r['country']['value'],
             'year': int(r['date']), 'life_expectancy': r['value'],
             'indicator_id': r['indicator']['id'], 'indicator_name': r['indicator']['value'],
             'unit': r['unit'], 'obs_status': r['obs_status']} for r in records]
    df = pd.DataFrame(rows)
    if len(df) < 300 or df.duplicated(['iso_alpha','country_worldbank','year']).any():
        raise RuntimeError('Invalid crawl: fewer than 300 records or duplicate keys')
    raw = ROOT/'data/raw'; raw.mkdir(parents=True, exist_ok=True)
    data = json.dumps([spider.meta, records], ensure_ascii=False).encode('utf-8')
    (raw/'worldbank_life_expectancy.json').write_bytes(data)
    df.to_csv(raw/'worldbank_scrapy_records.csv', index=False)
    audit = {'source_url': URL, 'tool': 'Scrapy', 'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
             'records': len(df), 'attributes': list(df.columns), 'pages': sorted(spider.pages),
             'sha256': hashlib.sha256(data).hexdigest(), 'mode': 'live_http',
             'http_200': crawler.stats.get_value('downloader/response_status_count/200', 0)}
    (ROOT/'reports/crawl_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(f'SCRAPY PASS: {len(df)} records, {len(df.columns)} fields')


if __name__ == '__main__':
    main()

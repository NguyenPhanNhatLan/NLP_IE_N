import scrapy


class RetinolSpider(scrapy.Spider):
    name = "retinol"
    
    def start_requests(self):
            yield scrapy.Request(
            url = "https://paulaschoice.vn/blogs/retinol/retinol-la-gi-thong-tin-ve-retinol?srsltid=AfmBOorMdcJrQ7hOiwDrdtl_TegQ2dTGGqeIIlw5zmKvsCXjAZfPwOyQ",
            callback = self.parse_dispatch
            
        )
        
    def parse_dispatch(self, response):
        data = response.xpath('//div[@class="article__body rte"]//p/text()').getall()
        text = " ".join(t.strip() for t in data if t.strip())
        return {
            "data": text
        }

import scrapy


class AlphaArbutinSpider(scrapy.Spider):
    name = "alpha_arbutin"
    
    def start_requests(self):
            yield scrapy.Request(
            url = "https://paulaschoice.vn/blogs/thanh-phan-my-pham/arbutin-la-gi-chi-ro-cong-dung-cua-arbutin-voi-lan-da",
            callback = self.parse_dispatch
            
        )
        
    def parse_dispatch(self, response):
        data = response.xpath('//div[@class="article__body rte"]//p/text()').getall()
        text = " ".join(t.strip() for t in data if t.strip())
        return {
            "data": text
        }

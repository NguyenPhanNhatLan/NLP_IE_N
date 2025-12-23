import scrapy


class TranexamicAcidSpider(scrapy.Spider):
    name = "tranexamic_acid"
        
    def start_requests(self):
            yield scrapy.Request(
            url = "https://paulaschoice.vn/blogs/tranexamic-acid/tranexamic-acid-co-tac-dung-gi-doi-voi-lan-da",
            meta={"playwright": True},
            callback = self.parse_dispatch
            
        )
        
    def parse_dispatch(self, response):
        data = response.xpath('//div[@class="article__body rte"]//p/text()').getall()
        text = " ".join(t.strip() for t in data if t.strip())
        return {
            "data": text
        }
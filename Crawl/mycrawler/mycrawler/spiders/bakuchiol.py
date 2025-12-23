import scrapy


class BakuchiolSpider(scrapy.Spider):
    name = "bakuchiol"
    def start_requests(self):
            yield scrapy.Request(
            url = "https://paulaschoice.vn/blogs/bakuchiol/cham-soc-da-lao-hoa-voi-bakuchiol",
            callback = self.parse_dispatch
            
        )
        
    def parse_dispatch(self, response):
        data = response.xpath('//div[@class="article__body rte"]//p/text()').getall()
        text = " ".join(t.strip() for t in data if t.strip())
        return {
            "data": text
        }
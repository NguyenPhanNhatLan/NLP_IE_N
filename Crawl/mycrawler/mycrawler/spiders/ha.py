import scrapy


class HaSpider(scrapy.Spider):
    name = "ha"
    def start_requests(self):
            yield scrapy.Request(
            url = "https://paulaschoice.vn/blogs/hyaluronic-acid-ha/hyaluronic-acid-la-gi-cong-dung-cua-hyaluronic-acid-voi-da",
            callback = self.parse_dispatch
            
        )
        
    def parse_dispatch(self, response):
        data = response.xpath('//div[@class="article__body rte"]//p/text()').getall()
        text = " ".join(t.strip() for t in data if t.strip())
        return {
            "data": text
        }

import scrapy


class GlycerinSpider(scrapy.Spider):
    name = "glycerin"
    def start_requests(self):
            yield scrapy.Request(
            url = "https://paulaschoice.vn/blogs/kien-thuc-cham-soc-da/glycerin-la-gi-cong-dung-cua-glycerin-trong-lam-dep",
            callback = self.parse_dispatch
            
        )
        
    def parse_dispatch(self, response):
        data = response.xpath('//div[@class="article__body rte"]//p/text()').getall()
        text = " ".join(t.strip() for t in data if t.strip())
        return {
            "data": text
        }

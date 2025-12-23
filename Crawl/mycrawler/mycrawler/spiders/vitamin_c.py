import scrapy


class VitaminCSpider(scrapy.Spider):
    name = "vitamin_c"
    
    def start_requests(self):
            yield scrapy.Request(
            url = "https://paulaschoice.vn/blogs/vitamin-c/tac-dung-cua-vitamin-c-voi-da",
            callback = self.parse_dispatch
            
        )
        
    def parse_dispatch(self, response):
        data = response.xpath('//div[@class="article__body rte"]//p/text()').getall()
        text = " ".join(t.strip() for t in data if t.strip())
        print(text)
        yield {
            "data": text
        }

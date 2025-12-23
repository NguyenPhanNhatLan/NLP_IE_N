import scrapy


class NiacinamideSpider(scrapy.Spider):
    name = "niacinamide"

    def start_requests(self):
        yield scrapy.Request(
            url = "https://paulaschoice.vn/blogs/niacinamide/niacinamide-la-gi-cong-dung-cua-niacinamide-trong-lam-dep?srsltid=AfmBOopjWGk5vm8L_wtod4llgUEZYmO-XnCqvtZjXhuu7EpFUD4sBkXK",
            callback = self.parse_dispatch
            
        )
        
    def parse_dispatch(self, response):
        data = response.xpath('//div[@class="article__body rte"]//p/text()').getall()
        text = " ".join(t.strip() for t in data if t.strip())
        return {
            "data" : text
        }
        
        
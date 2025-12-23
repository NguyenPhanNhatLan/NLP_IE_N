import re
import unicodedata
from typing import List

# python -m app.services.normalization
def normalize(text: str) -> List[str]:

    if not text or not text.strip():
        return []

    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"[\x00-\x09\x0b-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text)

    sentences = re.split(r"(?<=[.!?])\s+", text)

    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    return sentences

if __name__ == "__main__":
    text = 'Vitamin E (Tocopherol) là "thần dược" cho da, giúp dưỡng ẩm sâu, chống oxy hóa mạnh mẽ chống lão hóa, ' \
    'làm mờ thâm sạm, sẹo, tăng cường collagen, bảo vệ da khỏi tia UV và hỗ trợ phục hồi da cháy nắng, ' \
    'mang lại làn da mềm mịn, đàn hồi. Có thể dùng trực tiếp dầu từ viên nang (da khô) ' \
    'hoặc dùng sản phẩm có chứa Vitamin E (da dầu), kết hợp ăn uống thực phẩm giàu Vitamin E ' \
    '(hạnh nhân, bơ, rau xanh) để có làn da khỏe mạnh toàn diện. '

    test = normalize(text)
    print(test)


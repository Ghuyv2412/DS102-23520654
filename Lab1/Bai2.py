import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from datetime import datetime
from urllib.parse import urljoin

BASE_URL = "https://xemthoitiet.vn/"

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/120.0 Safari/537.36"
    )
}

VALID_LOCATIONS = {
    "Hà Giang",
    "Cao Bằng",
    "Bắc Kạn",
    "Tuyên Quang",
    "Thái Nguyên",
    "Lạng Sơn",
    "Quảng Ninh",
    "Bắc Giang",
    "Phú Thọ",
    "Lào Cai",
    "Điện Biên",
    "Lai Châu",
    "Sơn La",
    "Yên Bái",
    "Hòa Bình",
    "Hà Nội",
    "Vĩnh Phúc",
    "Bắc Ninh",
    "Hải Dương",
    "Hải Phòng",
    "Hưng Yên",
    "Thái Bình",
    "Hà Nam",
    "Nam Định",
    "Ninh Bình",
    "Thanh Hóa",
    "Nghệ An",
    "Hà Tĩnh",
    "Quảng Bình",
    "Quảng Trị",
    "Thừa Thiên Huế",
    "Đà Nẵng",
    "Quảng Nam",
    "Quảng Ngãi",
    "Bình Định",
    "Phú Yên",
    "Khánh Hòa",
    "Ninh Thuận",
    "Bình Thuận",
    "Kon Tum",
    "Gia Lai",
    "Đắk Lắk",
    "Đắk Nông",
    "Lâm Đồng",
    "Hồ Chí Minh",
    "Bà Rịa - Vũng Tàu",
    "Bình Phước",
    "Bình Dương",
    "Tây Ninh",
    "Đồng Nai",
    "An Giang",
    "Bạc Liêu",
    "Bến Tre",
    "Cà Mau",
    "Cần Thơ",
    "Đồng Tháp",
    "Hậu Giang",
    "Kiên Giang",
    "Long An",
    "Sóc Trăng",
    "Tiền Giang",
    "Trà Vinh",
    "Vĩnh Long"
}

def get_locations():
    response = requests.get(
        BASE_URL,
        headers=headers,
        timeout=20
    )

    if response.status_code != 200:
        return {}

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    locations = {}

    for a in soup.find_all("a", href=True):
        href = a["href"]
        name = a.get_text(
            " ",
            strip=True
        )
        if re.fullmatch(
            r"/thoi-tiet/[^/]+/?",
            href
        ):
            if name in VALID_LOCATIONS:

                locations[name] = urljoin(
                    BASE_URL,
                    href
                )
    return locations


def crawl_weather(location, base_url):
    url = base_url.rstrip("/") + "/7-ngay-toi/"
    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

    except requests.RequestException:
        return []
    if response.status_code != 200:
        return []
    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )
    text = soup.get_text(
        separator=" ",
        strip=True
    )
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    start = text.find("Dự báo thời tiết")
    end = text.find("Nhiệt độ và khả năng có mưa")

    if start == -1 or end == -1:
        return []

    weather_text = text[start:end]

    pattern = r"""
    (Hôm\ nay|T[2-7]\s+\d{2}/\d{2}|CN\s+\d{2}/\d{2})
    \s+
    (\d+)°\s*/\s*(\d+)°
    \s+
    (.+?)
    \s+
    (\d+)\s*%
    \s+
    ([\d.]+)\s*km/h
    \s+
    Ngày/đêm
    \s+
    (\d+)°/(\d+)°
    \s+
    Sáng/tối
    \s+
    (\d+)°/(\d+)°
    \s+
    Áp\ suất
    \s+
    (\d+)\s*hPa
    \s+
    Mặt\ trời\ mọc\ lặn
    \s+
    ([0-9:]+\s*[ap]m)
    \s*/\s*
    ([0-9:]+\s*[ap]m)
    \s+
    Độ\ ẩm
    \s+
    (\d+)%
    \s+
    Gió
    \s+
    ([\d.]+)\s*km/h
    """

    matches = re.findall(
        pattern,
        weather_text,
        re.VERBOSE | re.IGNORECASE
    )

    data = []

    crawl_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    for item in matches:

        data.append({
            "location": location,
            "date": item[0],
            "temp_min_c": int(item[1]),
            "temp_max_c": int(item[2]),
            "condition": item[3].strip(),
            "day_temp_c": int(item[6]),
            "night_temp_c": int(item[7]),
            "morning_temp_c": int(item[8]),
            "evening_temp_c": int(item[9]),
            "pressure_hpa": int(item[10]),
            "sunrise": item[11],
            "sunset": item[12],
            "humidity_percent": int(item[13]),
            "wind_speed_kmh": float(item[14]),
            "source_url": url,
            "crawl_time": crawl_time
        })

    return data


locations = get_locations()
all_data = []

for name, url in locations.items():
    weather_data = crawl_weather(
        name,
        url
    )
    all_data.extend(
        weather_data
    )
    time.sleep(1)


df = pd.DataFrame(
    all_data
)


if not df.empty:
    df.to_csv(
        "weather_data.csv",
        index=False,
        encoding="utf-8-sig"
    )
    df.to_csv(
        "weather_data.tsv",
        sep="\t",
        index=False,
        encoding="utf-8-sig"
    )
    df.to_json(
        "weather_data.json",
        orient="records",
        force_ascii=False,
        indent=4
    )

    print("Completed.")
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from urllib.parse import urljoin
BASE_URL = "https://xemthoitiet.vn/"
headers = {
    "User-Agent": "Mozilla/5.0"
}

def get_locations():
    response = requests.get(BASE_URL, headers=headers)
    if response.status_code != 200:
        print("NOT FOUND!")
        return {}
    soup = BeautifulSoup(response.text, "html.parser")
    locations = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        name = a.get_text(" ", strip=True)
        if re.fullmatch(r"/thoi-tiet/[^/]+/?", href):
            if (
                name != ""
                and "Hôm nay" not in name
                and "arrow_circle_right" not in name
                and "Thời tiết" not in name
                and len(name) < 30
            ):
                locations[name] = urljoin(BASE_URL, href)
    return locations

def crawl_weather(location, base_url):
    url = base_url.rstrip("/") + "/7-ngay-toi/"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    start = text.find("Dự báo thời tiết")
    end = text.find("Nhiệt độ và khả năng có mưa")
    if start == -1 or end == -1:
        return []

    weather_text = text[start:end]
    pattern = (
        r"(Hôm nay|T[2-7]\s+\d{2}/\d{2}|CN\s+\d{2}/\d{2})\s+"
        r"(\d+)°\s*/\s*(\d+)°\s+"
        r"(.+?)\s+(\d+)\s*%\s+([\d.]+)\s*km/h\s+"
        r"Ngày/đêm\s+(\d+)°/(\d+)°\s+"
        r"Sáng/tối\s+(\d+)°/(\d+)°\s+"
        r"Áp suất\s+(\d+)\s*hPa\s+"
        r"Mặt trời mọc lặn\s+"
        r"([0-9:]+\s*[ap]m)\s*/\s*([0-9:]+\s*[ap]m)\s+"
        r"Độ ẩm\s+(\d+)%\s+"
        r"Gió\s+([\d.]+)\s*km/h"
    )

    matches = re.findall(
        pattern,
        weather_text,
        re.IGNORECASE
    )

    data = []

    for item in matches:
        row = {
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
            "wind_speed_kmh": float(item[14])
        }
        data.append(row)
    return data

locations = get_locations()
print("Số địa điểm tìm được:", len(locations))
all_data = []
for name, url in locations.items():
    print("Đang lấy dữ liệu:", name)
    weather_data = crawl_weather(name, url)
    all_data.extend(weather_data)
    time.sleep(1)

df = pd.DataFrame(all_data)
print(df.head())

if not df.empty:
    df.to_csv(
        "weather_data.csv",
        index=False,
        encoding="utf-8-sig"
    )
    df.to_json(
        "weather_data.json",
        orient="records",
        force_ascii=False,
        indent=4
    )
    print("DONE.")
else:
    print("ERROR")
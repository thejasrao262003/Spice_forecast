from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

try:
    year = 2019
    for i in range(5):
        URL = "https://agmarknet.gov.in/"
        driver.get(URL)
        dropdown = driver.find_element(By.NAME, "ddlCommodity")
        select = Select(dropdown)
        select.select_by_visible_text("Sesamum(Sesame,Gingelly,Til)")

        from_date_input = driver.find_element(By.NAME, "txtDate")
        from_date_input.clear()
        from_date_input.send_keys(f"31-Oct-{year}")

        to_date_input = driver.find_element(By.NAME, "txtDateTo")
        to_date_input.clear()
        to_date_input.send_keys(f"31-Oct-{year+1}")
        go_button = driver.find_element(By.NAME, "btnGo")
        go_button.click()
        time.sleep(10)
        new_page_source = driver.page_source
        soup = BeautifulSoup(new_page_source, 'html.parser')
        table = soup.find("table", {"class": "tableagmark_new"})
        headers = [th.get_text(strip=True) for th in table.find_all("th")]

        rows = []
        for row in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if cells:
                rows.append(cells)

        df = pd.DataFrame(rows, columns=headers)
        df.to_csv(f"Data/spice_price_data_{year}.csv")
        year+=1

except Exception as e:
    print("Error:", e)

finally:
    driver.quit()

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

# Selenium setup
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

try:
    # Base URL
    URL = "https://agmarknet.gov.in/"
    driver.get(URL)
    # Loop through years
    for state in ["Karnataka", "Madhya Pradesh", "Uttar Pradesh", "Telangana", "Gujarat"]:
        time.sleep(8)
        from_date = f"26-Nov-2019"
        to_date = f"26-Nov-2024"

        def select_dropdown(select_name, option_text):
            while True:
                try:
                    dropdown = driver.find_element(By.NAME, select_name)
                    select = Select(dropdown)
                    select.select_by_visible_text(option_text)
                    break  # Break loop once successful
                except StaleElementReferenceException:
                    print(f"Retrying to select '{option_text}' for '{select_name}'...")

        # Select "Both" in arrival price dropdown
        select_dropdown("ddlArrivalPrice", "Both")
        dropdown2 = driver.find_element(By.ID, "ddlState")
        select = Select(dropdown2)
        select.select_by_visible_text(state)
        time.sleep(1)
        # Select the commodity
        select_dropdown("ddlCommodity", "Sesamum(Sesame,Gingelly,Til)")

        # Wait and enter from and to dates
        from_date_input = driver.find_element(By.NAME, "txtDate")
        from_date_input.clear()
        from_date_input.send_keys(from_date)

        to_date_input = driver.find_element(By.NAME, "txtDateTo")
        to_date_input.clear()
        to_date_input.send_keys(to_date)

        # Click Go button
        go_button = driver.find_element(By.NAME, "btnGo")
        go_button.click()
        time.sleep(5)  # Fixed wait after clicking Go

        # Parse the table data
        new_page_source = driver.page_source
        soup = BeautifulSoup(new_page_source, 'html.parser')
        table = soup.find("table", {"class": "tableagmark_new"})

        if table:
            # Extract headers and rows
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            rows = []
            for row in table.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if cells:
                    rows.append(cells)

            # Save data to a CSV file
            df = pd.DataFrame(rows, columns=headers)
            file_name = f"Data/{state}.csv"
            df.to_csv(file_name, index=False)
            print(f"Data saved to {file_name}")
        else:
            print(f"No data found")
        driver.get(URL)
        time.sleep(5)

except Exception as e:
    print("Error:", e)

finally:
    driver.quit()

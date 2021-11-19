import time

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

driver = webdriver.Firefox()
driver.get("https://www.youtube.com/watch?v=0QNiZfSsPc0")

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# # won't work unless you are logged in
# like_btn = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH,"(//yt-icon[@class='style-scope ytd-toggle-button-renderer'])[4]")))
# like_btn.click()

# # won't work unless you are logged in
# dislike_btn = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH,"(//yt-icon[@class='style-scope ytd-toggle-button-renderer'])[5]")))
# dislike_btn.click()
#
# pause_btn = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH,"//button[@title='Pause (k)']")))
# pause_btn.click()

# comment out to test pause btn, otherwise it happens so fast you don't notice
play_btn = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH,"//button[@title='Play (k)']")))
play_btn.click()

# mute_btn = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH,"//button[@aria-label='Mute (m)']")))
# mute_btn.click()
time.sleep(5)
driver.execute_script("window.scrollTo(0, 700)")
# comment out to test mute_btn, otherwise it happens so fast you don't notice it
# unmute_btn = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH,"//button[@aria-label='Unmute (m)']")))
# unmute_btn.click()
# driver.close()

import os
import requests
import sys
from bs4 import BeautifulSoup as soup

def geturl(url):
    str1 = requests.get(url).text
    str2 = soup(str1)
    res = ""
    for x in str2.find_all("img"):
        y = str(x.attrs['src'])
        if y.find("96") != -1:
            res = res + y + '\n'
    with open("url.txt", 'a') as file:
        file.write(res)

for lines in sys.stdin:
    geturl(lines)

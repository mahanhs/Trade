
import urllib.request
url = "https://api.nomics.com/v1/exchange-rates?key=6dc7b710408687f9f920763c962a2f6650f178b9"
print(urllib.request.urlopen(url).read())
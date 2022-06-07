import requests
import base64
import json
import cv2
import time

URL_BASE = 'https://beerscan-image-wkgvoiogvq-ew.a.run.app'
URL_IDENTIFY_BEERS = URL_BASE + '/identify_beers'

def test_boxes_endpoint(img_b64):

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

    payload = json.dumps({"image": img_b64})
    response = requests.post(URL_IDENTIFY_BEERS, data=payload, headers=headers)

    try:
        data = response.json()

    except requests.exceptions.RequestException:
        print(response.text)

    return data

def main():

    image_file = 'raw_data/images/test_img/belgian_beer_tour.jpg'

    with open(image_file, "rb") as f:
        im_bytes = f.read()
    img_b64 = base64.b64encode(im_bytes).decode("utf8")
    data = test_boxes_endpoint(img_b64)

    print(data)


if __name__ == '__main__':
    main()

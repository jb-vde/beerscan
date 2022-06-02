import requests
import base64
import json
import cv2

URL_BASE = 'https://beerscan-image-wkgvoiogvq-ew.a.run.app'
URL_BOXES = URL_BASE + '/predict_boxes'
URL_SIZE = URL_BASE + '/size'


def print_image(image, boxes):
    color = (0, 255, 0)
    for box in boxes.values():
        cv2.rectangle(image, (box['startX'], box['startY']),
                      (box['endX'], box['endY']), color, 2)
    cv2.imshow("Output", image)
    cv2.waitKey(0)


def test_base_endpoint():
    response = requests.get(URL_BASE)
    print('Base endpoint:')
    print(response.json())


def test_size_endpoint(img_b64):
    files={"file": img_b64}
    response = requests.post(URL_SIZE, data=files)
    print('Size endpoint:')
    print(response.json())


def test_boxes_endpoint(img_b64):

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

    payload = json.dumps({"image": img_b64})
    response = requests.post(URL_BOXES, data=payload, headers=headers)

    try:
        data = response.json()
        print(data)
    except requests.exceptions.RequestException:
        print(response.text)

    return data

def main():

    image_file = 'raw_data/images/test_amis.jpg'
    image = cv2.imread(image_file)

    with open(image_file, "rb") as f:
        im_bytes = f.read()
    img_b64 = base64.b64encode(im_bytes).decode("utf8")

    test_base_endpoint
    test_size_endpoint(img_b64)
    data = test_boxes_endpoint(img_b64)


    print_image(image, data['boxes'])


if __name__ == '__main__':
    main()

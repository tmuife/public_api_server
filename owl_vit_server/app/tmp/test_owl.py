import asyncio
from PIL import Image, ImageDraw
from io import BytesIO
import base64
from app.services.owl_vit_service import OWLVIT

def imageToBase64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_data

async def detect():
    owl = OWLVIT()
    image = Image.open("/Users/walter/Downloads/test.jpg")
    drawn_image = image.copy()
    image_base = imageToBase64(image)
    texts = "table| chair | face"
    result = owl.get_inference(image_base, texts)
    print(result)

    draw = ImageDraw.Draw(drawn_image)
    # 绘制边框
    try:
        for item in result:
            box = item["location"]
            label = item["label"]
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1] - 10), label, fill="red")
        drawn_image.show()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    #asyncio.run(convert_video_video())
    asyncio.run(detect())
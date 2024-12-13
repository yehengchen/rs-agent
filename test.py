from PIL import Image

def main():
    # 打开图片
    image = Image.open('/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/image/20241015_171211/9e01287f.png')

    # 获取图片的宽度和高度
    width, height = image.size

    # 打印图片的宽度和高度
    print(f'图片的宽度为：{(image)}')
    print(f'图片的宽度为：{width}')
    print(f'图片的高度为：{height}')


def process_image(image_path)->Image:
    # 打开图片
    image = Image.open(image_path)

    # 获取图片的宽度和高度
    width, height = image.size

    # 打印图片的宽度和高度
    print(f'图片的宽度为：{width}')
    print(f'图片的高度为：{height}')
    return image
    
if __name__ == '__main__':
    image_path = '/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/image/20241015_171211/9e01287f.png'
    process_image(image_path)
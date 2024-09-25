# from RStask.ObjectDetection.YOLOv5 import YoloDetection
from YOLOv5 import YoloDetection

def run_detection(image_path ,updated_image_path):
    
    model=YoloDetection('cuda:0')
    det=model.inference_app(image_path, updated_image_path)
    return det

if __name__ == "__main__":

    image_path = '/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/image/rs-ship.jpg'
    updated_image_path = '/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/image/'
    run_detection(image_path ,updated_image_path)
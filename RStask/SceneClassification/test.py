from ResNetScene import ResNetAID

def run_classification(image_path):
    
    model=ResNetAID('cuda:0')
    cls=model.inference(image_path)
    return cls

if __name__ == "__main__":

    image_path = '/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/image/rs-ship.jpg'
    cls_rslt = run_classification(image_path)
    print(cls_rslt)



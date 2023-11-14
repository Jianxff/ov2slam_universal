import cv2
import sys
import ov2slam
import time


def load_image(img_folder, img_ts_path):
  imgs = []
  ts = []
  
  with open(img_ts_path, 'r') as f:
    ts = f.read().strip().split()
  
  for(_, t) in enumerate(ts):
    img_path = img_folder + '/' + t + '.png'
    imgs.append(img_path)
  
  return imgs, ts

# python mono_euroc.py parameters_files/accurate/euroc/euroc_mono.yaml ~/dataset/mav0/cam0/data euroc_ts/V101.txt

if __name__ == '__main__' :
  conf, img_folder, img_ts = sys.argv[1], sys.argv[2], sys.argv[3]
  
  imgs, ts = load_image(img_folder, img_ts)
  
  session = ov2slam.Session.create(conf)
  session.startVisualize()
  
  input('wait for pangolin')
  
  for i in range(0, len(imgs)):
    img = cv2.imread(imgs[i])
    session.addTrack(img, float(ts[i]))
    time.sleep(1/30)
    
  input()
  session.stop()
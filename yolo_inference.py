from ultralytics import YOLO 

model = YOLO ('models/best_ylv8_ep50.pt')

results = model.predict('input_videos/cut30s.mp4', conf=0.5, save=True)
print(results[0])
print('================================')
for box in results[0].boxes:
    print(box)

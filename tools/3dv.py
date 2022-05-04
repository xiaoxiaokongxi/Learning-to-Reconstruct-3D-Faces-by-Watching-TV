from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, extract_face

path = "/cluster/home/jiezcao/face_recognition/test/"
img = Image.open(path+'friend.jpg') 
mtcnn = MTCNN(keep_all=True)

boxes, probs, points = mtcnn.detect(img, landmarks=True)
print(probs)
# Draw boxes and save faces
if probs.all():
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    for i, (box, point) in enumerate(zip(boxes, points)):
        draw.rectangle(box.tolist(), width=5)
        for p in point:
            draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
            extract_face(img, box, save_path=path+'/detected_face_{}.png'.format(i))
        img_draw.save(path+'annotated_faces.png')
else:
    print("Cannot detect face!!!")

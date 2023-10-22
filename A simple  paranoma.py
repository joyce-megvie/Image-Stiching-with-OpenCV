import cv2 

# load the images to be stitched
images = ['7.jpg','8.jpg'] 
imgs = [] 
  
for i in range(len(images)): 
    imgs.append(cv2.imread(images[i])) 
    imgs[i]=cv2.resize(imgs[i],(0,0),fx=0.4,fy=0.4) 
    
# showing the original pictures 
cv2.imshow('Image1',imgs[0]) 
cv2.imshow('Image2',imgs[1]) 

stitchy=cv2.Stitcher.create() 
(dummy,output)=stitchy.stitch(imgs) 
  
if dummy != cv2.STITCHER_OK: 
    print("stitching ain't successful") 
else:  
    print('Your Panorama is ready!!!') 

cv2.imshow('final result', output)
cv2.waitKey(0)
from PIL import Image

import  numpy as np

img = Image.open('/usr/stud/george/test.png')
arr = np.asarray(img)
#img.show()
p = img.palette

save_loc = '/usr/stud/george/test_write.png'
#img2 = Image.
#img2.save(save_loc)

img3 = Image.open(save_loc)
#img3.palette = p
p1 = img.getpalette()
img3.putpalette(p1)
#img3.quantize(method = 1)
#img3.convert('P',palette = Image.ADAPTIVE,colors=8)
#img3.convert('P',palette = Image.ADAPTIVE,colors=256)
img3.save(save_loc)

a = np.ones((100,100),dtype=np.uint8)
for i in range(100):
    a[i,:] = i

img4 = Image.fromarray(a,mode='P')
img4.putpalette(p1)
img4.show()
"""
	resize an image to required dimensions
"""

from PIL import Image

from resizeimage import resizeimage

file_no = 22
# x_train = tf.placeholder(float32, [None, 32256])
# y_train = tf.palceholder(float32, [None, file_no1-1])
labels= []
images= []
for i in xrange(18,file_no+1):
	if i==14: continue
	nmbr = str(i)
	dirc = 'yaleB' + nmbr
	for j in xrange(0,9):
		nmbr2='0'+str(j)
		file_info = open(dirc+'/'+dirc+'_P'+nmbr2+'.info','r')
		#label = np.zeros(2)

		for line in file_info.readlines():
			line = dirc+'/'+line.strip()
			print line
			with open(line, 'r+b') as f:
			    with Image.open(f) as image:
				cover = resizeimage.resize_cover(image, [168, 192])
				cover.save('images/'+line, image.format)

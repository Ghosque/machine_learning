from PIL import Image


"""旋转后图片质量降低"""
path = r'E:\Python\Ghosque\data\pillow_test.jpg'
rotate_path = r'E:\Python\Ghosque\data\pillow_test_rotate.jpg'
transpose_path = r'E:\Python\Ghosque\data\pillow_testtranspose.jpg'
im = Image.open(path)
im.rotate(6).save(rotate_path)

im.transpose(Image.FLIP_LEFT_RIGHT).save(transpose_path)





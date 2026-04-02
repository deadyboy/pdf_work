from PIL import Image

# 加载图片
img_path = '/data1/jianf/新提取pdf/data/ICU_Page_1_HighRes.png'  # 请确保图片名正确
img = Image.open(img_path)

# 获取图片尺寸
width, height = img.size

# 根据你的图片比例，“给氧方式”大约位于横向 21% 的位置
# 建议手动微调这个 split_point 像素值以达到最精确的效果
split_point = int(width * 0.4) 

# 进行裁剪 (左, 上, 右, 下)
left_half = img.crop((0, 0, split_point, height))

# 保存图片，quality=100 并保持 subsampling 以防画质损失
left_half.save('left_side_medical_record.jpg', subsampling=0, quality=100)

print(f"处理完成！左半部分已保存。切分点像素位置: {split_point}")
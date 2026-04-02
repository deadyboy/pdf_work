import os
import cv2
import shutil
from pathlib import Path
from paddleocr import PaddleOCR

def classify_icu_records(input_dir, output_base_dir):
    """
    通过读取图片表头的文本，自动分类《记录单一》和《记录单二》
    """
    # 1. 创建分类后的目标文件夹
    dir_record1 = os.path.join(output_base_dir, "data_record1")
    dir_record2 = os.path.join(output_base_dir, "data_record2")
    dir_unknown = os.path.join(output_base_dir, "data_unknown") # 存放识别失败的图，防遗漏
    
    os.makedirs(dir_record1, exist_ok=True)
    os.makedirs(dir_record2, exist_ok=True)
    os.makedirs(dir_unknown, exist_ok=True)

    # 获取所有图片
    input_path = Path(input_dir)
    image_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))
    
    if not image_files:
        print("未找到任何图片，请检查输入路径。")
        return

    print(f"🚀 找到 {len(image_files)} 张图片，启动 PaddleOCR 智能路由分类...")
    
    # 2. 初始化 OCR (仅需初始化一次)
    # 因为只是识别表头几个大字，不需要方向分类器，速度极快
    ocr = PaddleOCR(use_angle_cls=False, lang='ch', show_log=False)
    
    count_1 = 0
    count_2 = 0
    count_unk = 0

    # 3. 遍历分类
    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            H, W = img.shape[:2]
            
            # 核心优化：只裁剪顶部 15% 的区域进行 OCR，大幅提升速度并排除下方手写噪点干扰
            crop_h = int(H * 0.15)
            header_img = img[0:crop_h, :]

            # 执行 OCR 识别
            results = ocr.ocr(header_img, cls=False)
            text_str = ""
            if results and results[0]:
                for line in results[0]:
                    text_str += line[1][0]
            
            # 清洗字符串（兼容中文和英文括号，去除空格）
            text_str = text_str.replace(" ", "").replace("（", "(").replace("）", ")")

            # 4. 路由分发逻辑
            # 根据标题关键字判断
            if "单(一)" in text_str or "单(1)" in text_str:
                shutil.copy(img_path, os.path.join(dir_record1, img_path.name))
                count_1 += 1
                print(f"[单一] 分流成功: {img_path.name}")
                
            elif "单(二)" in text_str or "单(2)" in text_str:
                shutil.copy(img_path, os.path.join(dir_record2, img_path.name))
                count_2 += 1
                print(f"[单二] 分流成功: {img_path.name}")
                
            else:
                # 兜底逻辑：如果标题被污染，找专有列名
                if "神经系统" in text_str or "导管评估" in text_str:
                    shutil.copy(img_path, os.path.join(dir_record2, img_path.name))
                    count_2 += 1
                    print(f"[单二] (兜底) 分流成功: {img_path.name}")
                elif "入量" in text_str or "出量" in text_str:
                    shutil.copy(img_path, os.path.join(dir_record1, img_path.name))
                    count_1 += 1
                    print(f"[单一] (兜底) 分流成功: {img_path.name}")
                else:
                    # 彻底认不出来，放进 unknown 供人工复核
                    shutil.copy(img_path, os.path.join(dir_unknown, img_path.name))
                    count_unk += 1
                    print(f"[未知] ⚠️ 无法识别: {img_path.name}")
                    
        except Exception as e:
            print(f"处理图片 {img_path.name} 时发生错误: {e}")
            shutil.copy(img_path, os.path.join(dir_unknown, img_path.name))
            count_unk += 1

    print("\n" + "="*40)
    print("🎉 路由分发完成！")
    print(f"👉 记录单(一): {count_1} 张")
    print(f"👉 记录单(二): {count_2} 张")
    print(f"👉 需人工核查 (Unknown): {count_unk} 张")
    print("="*40)

if __name__ == "__main__":
    # 你的混合图片存放目录
    MIXED_INPUT_DIR = "/data1/jianf/新提取pdf/mixed_data_for_server" 
    
    # 分类后存放的总目录
    ROUTED_OUTPUT_DIR = "/data1/jianf/新提取pdf/routed_data" 
    
    classify_icu_records(MIXED_INPUT_DIR, ROUTED_OUTPUT_DIR)
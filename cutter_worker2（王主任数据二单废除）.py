# cutter_worker2.py
import cv2
import numpy as np
import os
import shutil
import re
import argparse
from paddleocr import PaddleOCR

TIME_PATTERN = re.compile(r"([01]?\d|2[0-3])[:：][0-5]\d")

def get_time_from_zone(img_segment, ocr_engine):
    try:
        if img_segment is None or img_segment.size == 0: return None
        result = ocr_engine.ocr(img_segment, cls=False)
        if not result or not result[0]: return None
        for line in result[0]:
            text = line[1][0].replace(" ", "")
            m = TIME_PATTERN.search(text)
            if m: return m.group(0).replace("：", ":")
    except Exception:
        return None
    return None

def get_global_splits_by_counting(header_img):
    """
    核心修改：只在表头区域全局数一次格子，精准计算 x1 和 x2
    """
    h, w = header_img.shape[:2]
    gray = cv2.cvtColor(header_img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15)
    
    # 只要是高度大于 30 像素的竖线，我们全都要（把表头里那些很短的子列线全抓出来）
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=2)
    contours, _ = cv2.findContours(v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    col_xs = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch > 30: col_xs.append(x + cw // 2)
            
    col_xs = sorted(list(set(col_xs)))
    if len(col_xs) > 0:
        merged_xs = [col_xs[0]]
        for v in col_xs[1:]:
            if v - merged_xs[-1] > 10: merged_xs.append(v)
        col_xs = merged_xs

    print(f"🔍 [Worker] 在表头区精确识别到 {len(col_xs)} 根网格竖线")
    
    # 记录单二的标准列数通常是 50~52 根
    if len(col_xs) >= 45: 
            # 🌟 核心修改 1：你需要自己重新数一下，三刀分别要切在第几根竖线上
            # 比如：第一刀在第12根，第二刀在第26根，第三刀在第40根 (请根据实际情况修改下面的数字)
            x1, x2, x3 = col_xs[21], col_xs[38], col_xs[48]
            print(f"✅ [Worker] 按网格数成功定位切分坐标: x1={x1}, x2={x2}, x3={x3}")
    else:
            print(f"⚠️ [Worker] 竖线数量不足 ({len(col_xs)} 根)，启用智能比例兜底")
            # 如果找不到竖线，按宽度均分为四等分兜底
            target1, target2, target3 = int(w * 0.25), int(w * 0.50), int(w * 0.75)
            x1 = min(col_xs, key=lambda x: abs(x - target1)) if col_xs else target1
            x2 = min(col_xs, key=lambda x: abs(x - target2)) if col_xs else target2
            x3 = min(col_xs, key=lambda x: abs(x - target3)) if col_xs else target3

    coords = (max(0, min(x1, w-1)), max(x1, min(x2, w-1)), max(x2, min(x3, w-1)))        
    return coords, col_xs


def optimize_image_for_llm(img):
    # 1. 放大 2 倍 (双三次插值对文字最友好)
    h, w = img.shape[:2]
    enlarged = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    # 2. 四周补 40 像素的白边，防止边缘文字被模型吃掉
    padded = cv2.copyMakeBorder(enlarged, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return padded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default="/data1/jianf/新提取pdf/data2/1_11_ICU护理记录单_2020年09月24日_35241.png", help="输入图片路径")
    parser.add_argument("--out", default="icu_slices", help="输出文件夹")
    args = parser.parse_args()

    img_path = args.img
    output_base = args.out
    
    # 依然保持你之前的表头高度设定
    header_bottom_y = 543

    print(f"🔍 [Worker] 子进程启动，开始处理图片: {img_path}")

    if os.path.exists(output_base): shutil.rmtree(output_base)
    os.makedirs(output_base, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ [Worker] 致命错误：OpenCV 无法读取图片 {img_path}")
        return

    H, W = img.shape[:2]
    print(f"🔍 [Worker] 图片读取成功，分辨率: 宽{W} x 高{H}")

    ocr = PaddleOCR(use_angle_cls=False, lang='ch', show_log=False)
    print(f"🔍 [Worker] PaddleOCR 模型加载完成")

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 10, 1))
        h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=2)
        
        contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 提取横线
        all_y = sorted([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3]//2 for c in contours if cv2.boundingRect(c)[2] > W * 0.5])
        
        if not all_y: 
            return
            
        merged_y = [all_y[0]]
        for y in all_y[1:]:
            if y - merged_y[-1] > 15: merged_y.append(y)
            
    except Exception as e:
        print(f"❌ [Worker] 横线检测阶段发生异常: {e}")
        return

    # ======== 【核心修改位置】 ========
    # 先把表头截出来，只对表头算一次全局的 X 切分坐标！
    header = img[0:header_bottom_y, :]
    (global_x1, global_x2, global_x3), all_sub_lines = get_global_splits_by_counting(header)

    time_col_width = int(W * 0.15)
    block_starts = [] 
    
    print(f"🔍 [Worker] 正在扫描每一行的时间锚点...")
    for i in range(len(merged_y) - 1):
        y_top, y_btm = merged_y[i], merged_y[i+1]
        if y_top < header_bottom_y: continue
        try:
            time_zone = img[y_top:y_btm, :time_col_width]
            time_str = get_time_from_zone(time_zone, ocr)
            if time_str: block_starts.append((i, time_str))
        except Exception as e: 
            continue

    if not block_starts:
        return

    vis_img = img.copy()
    success_count = 0
    
    for idx in range(len(block_starts)):
        try:
            start_node_idx, t_str = block_starts[idx]
            y_start = merged_y[start_node_idx]
            y_end = merged_y[block_starts[idx+1][0]] if idx < len(block_starts) - 1 else merged_y[-1]

            final_block = np.vstack((header, img[y_start:y_end, :]))
            header_h = header.shape[0] # 表头高度
            for x_line in all_sub_lines:
                # 在拼接好的块上画 2 像素宽的红线 (BGR颜色: 0, 0, 255)
                cv2.line(final_block, (x_line, header_h), (x_line, final_block.shape[0]), (0, 0, 255), 5)
            # ======== 【核心修改位置】 ========
            # 直接使用全局计算好的坐标切分，速度极快且绝对统一
            part_1 = final_block[:, 0:global_x1]
            part_2 = final_block[:, global_x1:global_x2]
            part_3 = final_block[:, global_x2:global_x3]
            part_4 = final_block[:, global_x3:W]
            
            cv2.imwrite(os.path.join(output_base, f"block_{idx:02d}_1.png"), optimize_image_for_llm(part_1))
            cv2.imwrite(os.path.join(output_base, f"block_{idx:02d}_2.png"), optimize_image_for_llm(part_2))
            cv2.imwrite(os.path.join(output_base, f"block_{idx:02d}_3.png"), optimize_image_for_llm(part_3))
            cv2.imwrite(os.path.join(output_base, f"block_{idx:02d}_4.png"), optimize_image_for_llm(part_4))
            
            # 画上诊断线，方便预览
            cv2.rectangle(vis_img, (2, y_start), (W-2, y_end), (255, 0, 0), 2) 
            cv2.line(vis_img, (global_x1, y_start), (global_x1, y_end), (0, 0, 255), 3)      
            cv2.line(vis_img, (global_x2, y_start), (global_x2, y_end), (0, 255, 0), 3)  
            cv2.line(vis_img, (global_x3, y_start), (global_x3, y_end), (255, 0, 255), 3) # 画第三条切分线
            success_count += 1
        except Exception as e: 
            continue

    cv2.imwrite(os.path.join(output_base, "_block_preview.png"), vis_img)
    print(f"✅ [Worker] 图片切割顺利完成！共切出 {success_count} 行记录碎片。")

if __name__ == "__main__":
    main()
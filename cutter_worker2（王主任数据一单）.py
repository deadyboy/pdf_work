# cutter_worker1.py
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
    except Exception: return None
    return None

def get_global_splits_by_counting(header_img):
    h, w = header_img.shape[:2]
    gray = cv2.cvtColor(header_img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15)
    
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
    
    # 记录单(一)通常切两刀，分为：生命体征区、出入量区、长文本区
    if len(col_xs) >= 25: 
        # ⚠️ 需要人工校准：根据《记录单一》的竖线数量，修改这里的索引
        # 假设第12根线是生命体征右侧，第27根线是出量右侧
        x1, x2 = col_xs[14], col_xs[27] 
        print(f"✅ [Worker] 按网格数定位切分坐标: x1={x1}, x2={x2}")
    else:
        print(f"⚠️ [Worker] 竖线数量异常，启用兜底")
        target1, target2 = int(w * 0.33), int(w * 0.66)
        x1 = min(col_xs, key=lambda x: abs(x - target1)) if col_xs else target1
        x2 = min(col_xs, key=lambda x: abs(x - target2)) if col_xs else target2
    coords = max(0, min(x1, w-1)), max(x1, min(x2, w-1))
    return coords, col_xs

def optimize_image_for_llm(img):
    h, w = img.shape[:2]
    enlarged = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    padded = cv2.copyMakeBorder(enlarged, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return padded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default="/data1/jianf/新提取pdf/data3/1_1_ICU护理记录单_2020年09月24日_35241.png")
    parser.add_argument("--out", default="icu_slices")
    args = parser.parse_args()

    img_path, output_base = args.img, args.out
    
    # ⚠️ 需要人工校准：记录单一表头的像素高度，可能需要微调
    header_bottom_y = 600

    if os.path.exists(output_base): shutil.rmtree(output_base)
    os.makedirs(output_base, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None: return
    H, W = img.shape[:2]

    ocr = PaddleOCR(use_angle_cls=False, lang='ch', show_log=False)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 10, 1))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=2)
    contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_y = sorted([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3]//2 for c in contours if cv2.boundingRect(c)[2] > W * 0.5])
    if not all_y: return
        
    merged_y = [all_y[0]]
    for y in all_y[1:]:
        if y - merged_y[-1] > 15: merged_y.append(y)

    header = img[0:header_bottom_y, :]
    (global_x1, global_x2), all_sub_lines = get_global_splits_by_counting(header)

    time_col_width = int(W * 0.15)
    # 🌟 步骤 1：先扫描出所有真正带“时间”的主锚点
# ================= 🌟 核心修复区开始 =================

    # 1. 动态寻找“数据区”的真正起始线索引 (跳过表头内部的横线)
    start_data_idx = 0
    for i, y in enumerate(merged_y):
        # 允许 10 像素的误差容错
        if y >= header_bottom_y - 10: 
            start_data_idx = i
            break

    # 2. 扫描真正的时间锚点 (从数据区起始线开始扫)
    time_anchors = [] 
    for i in range(start_data_idx, len(merged_y) - 1):
        y_top, y_btm = merged_y[i], merged_y[i+1]
        try:
            time_zone = img[y_top:y_btm, :time_col_width]
            if get_time_from_zone(time_zone, ocr): 
                time_anchors.append(i) 
        except Exception: 
            continue

    # 3. 智能孤儿行回收补丁 (不再无脑插 0，而是插入真正的起跑线)
    block_starts = []
    if not time_anchors:
        # 极端情况：整页都没时间，从数据区第一条线切到底
        block_starts.append((start_data_idx, "")) 
    else:
        # 🌟 如果第一个时间锚点不在数据区第一行，说明顶部有孤儿行！
        if time_anchors[0] > start_data_idx:
            block_starts.append((start_data_idx, "")) # 完美回收：插入表头底线索引
            
        for idx in time_anchors:
            block_starts.append((idx, ""))

    # ================= 🌟 核心修复区结束 =================

    vis_img = img.copy()
    success_count = 0
    
    for idx in range(len(block_starts)):
        try:
            start_node_idx = block_starts[idx][0]
            y_start = merged_y[start_node_idx]
            y_end = merged_y[block_starts[idx+1][0]] if idx < len(block_starts) - 1 else merged_y[-1]

            final_block = np.vstack((header, img[y_start:y_end, :]))
            header_h = header.shape[0] 
            
            # 画辅助红线
            for x_line in all_sub_lines:
                cv2.line(final_block, (x_line, header_h), (x_line, final_block.shape[0]), (0, 0, 255), 4)
                
            part_1 = final_block[:, 0:global_x1]
            part_2 = final_block[:, global_x1:global_x2]
            part_3 = final_block[:, global_x2:W]
            
            cv2.imwrite(os.path.join(output_base, f"block_{idx:02d}_1.png"), optimize_image_for_llm(part_1))
            cv2.imwrite(os.path.join(output_base, f"block_{idx:02d}_2.png"), optimize_image_for_llm(part_2))
            cv2.imwrite(os.path.join(output_base, f"block_{idx:02d}_3.png"), optimize_image_for_llm(part_3))
            
            cv2.rectangle(vis_img, (2, y_start), (W-2, y_end), (255, 0, 0), 2) 
            cv2.line(vis_img, (global_x1, y_start), (global_x1, y_end), (0, 0, 255), 3)      
            cv2.line(vis_img, (global_x2, y_start), (global_x2, y_end), (0, 255, 0), 3)  
            success_count += 1
        except Exception: continue

    cv2.imwrite(os.path.join(output_base, "_block_preview.png"), vis_img)

if __name__ == "__main__":
    main()
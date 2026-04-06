# cutter_worker.py
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

# 🌟 新增：从表头一次性提取所有竖线坐标和全局切分点（移植自王主任一单 cutter）
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
    
    # 金主任数据切两刀分三部分：生命体征区(L)、出入量区(M)、护理文本区(R)
    if len(col_xs) >= 32: 
        x1, x2 = col_xs[20], col_xs[31] 
        print(f"✅ [Worker] 按网格数定位切分坐标: x1={x1}, x2={x2}")
    else:
        print(f"⚠️ [Worker] 竖线数量异常 ({len(col_xs)} 根)，启用比例兜底")
        target1, target2 = int(w * 0.41), int(w * 0.72)
        x1 = min(col_xs, key=lambda x: abs(x - target1)) if col_xs else target1
        x2 = min(col_xs, key=lambda x: abs(x - target2)) if col_xs else target2
    coords = max(0, min(x1, w-1)), max(x1, min(x2, w-1))
    return coords, col_xs

# 🌟 新增：图像优化 — 2倍放大+白边补边（移植自王主任 cutter）
def optimize_image_for_llm(img):
    h, w = img.shape[:2]
    enlarged = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    padded = cv2.copyMakeBorder(enlarged, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return padded

# 🌟 新增：保存切片图片+OCR文本到同名txt（移植自王主任二单 cutter）
def save_part_with_ocr(img_part, base_dir, name_prefix, ocr_engine):
    """
    保存切片图片，同时在原始分辨率下跑一次 OCR，并将文本保存为同名 txt
    """
    # 1. 跑 OCR 提取文本
    res = ocr_engine.ocr(img_part, cls=False)
    texts = []
    if res and res[0]:
        texts = [line[1][0] for line in res[0]]
    txt_str = ", ".join(texts)
    
    # 2. 保存为同名 txt 字典文件
    with open(os.path.join(base_dir, f"{name_prefix}.txt"), "w", encoding="utf-8") as f:
        f.write(txt_str)
        
    # 3. 保存给 LLM 看的放大补边优化版图片
    cv2.imwrite(os.path.join(base_dir, f"{name_prefix}.png"), optimize_image_for_llm(img_part))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="输入图片路径")
    parser.add_argument("--out", default="icu_slices", help="输出文件夹")
    args = parser.parse_args()

    img_path = args.img
    output_base = args.out
    # header_bottom_y = 526
    header_bottom_y = 430

    print(f"🔍 [Worker] 子进程启动，开始处理图片: {img_path}")

    if os.path.exists(output_base): shutil.rmtree(output_base)
    os.makedirs(output_base, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ [Worker] 致命错误：OpenCV 无法读取图片 {img_path}")
        return

    H, W = img.shape[:2]
    print(f"🔍 [Worker] 图片读取成功，分辨率: 宽{W} x 高{H}")

    # 初始化 OCR
    ocr = PaddleOCR(use_angle_cls=False, lang='ch', show_log=False)
    print(f"🔍 [Worker] PaddleOCR 模型加载完成")

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 10, 1))
        h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=2)
        
        contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 【重要放宽】：只要横线长度超过宽度的 50% 就认（原为 70%）
        all_y = sorted([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3]//2 for c in contours if cv2.boundingRect(c)[2] > W * 0.5])
        
        if not all_y: 
            print(f"❌ [Worker] 错误：未检测到有效横线 (长度>50%宽度)！")
            # 存下二值化线条图，看看究竟是什么干扰了 OpenCV
            debug_path = os.path.join(output_base, "debug_h_lines_failed.png")
            cv2.imwrite(debug_path, h_lines)
            print(f"📸 [Worker] 已将横线提取的诊断图保存至: {debug_path}，请检查。")
            return
            
        print(f"🔍 [Worker] 检测到 {len(all_y)} 条初筛横线...")
            
        merged_y = [all_y[0]]
        for y in all_y[1:]:
            if y - merged_y[-1] > 15: merged_y.append(y)
            
        print(f"🔍 [Worker] 合并后剩余 {len(merged_y)} 条有效横行边界")
            
    except Exception as e:
        print(f"❌ [Worker] 横线检测阶段发生异常: {e}")
        return

    # 🌟 改进：先把表头截出来，只对表头算一次全局的 X 切分坐标（替代原来每块独立算的 split_into_three_columns）
    header = img[0:header_bottom_y, :]
    (global_x1, global_x2), all_sub_lines = get_global_splits_by_counting(header)

    # 🌟 新增：截取患者信息区（移植自王主任二单 cutter）
    header_info_zone = img[0:300, :]
    cv2.imwrite(os.path.join(output_base, "_header_info.png"), optimize_image_for_llm(header_info_zone))

    time_col_width = int(W * 0.15)
    
    # ================= 🌟 智能孤儿行回收（移植自王主任一单 cutter） =================

    # 1. 动态寻找"数据区"的真正起始线索引 (跳过表头内部的横线)
    start_data_idx = 0
    for i, y in enumerate(merged_y):
        if y >= header_bottom_y - 10: 
            start_data_idx = i
            break

    # 2. 扫描真正的时间锚点 (从数据区起始线开始扫)
    print(f"🔍 [Worker] 正在扫描每一行的时间锚点...")
    time_anchors = [] 
    for i in range(start_data_idx, len(merged_y) - 1):
        y_top, y_btm = merged_y[i], merged_y[i+1]
        try:
            time_zone = img[y_top:y_btm, :time_col_width]
            if get_time_from_zone(time_zone, ocr): 
                time_anchors.append(i) 
        except Exception as e: 
            print(f"⚠️ [Worker] 扫描第 {i} 行时间区域时出错: {e}")
            continue

    # 3. 智能孤儿行回收补丁 (不再无脑插 0，而是插入真正的起跑线)
    block_starts = []
    if not time_anchors:
        # 极端情况：整页都没时间，从数据区第一条线切到底
        block_starts.append((start_data_idx, ""))
        print(f"⚠️ [Worker] 未找到时间锚点，将数据区整体作为一个块处理")
    else:
        # 🌟 如果第一个时间锚点不在数据区第一行，说明顶部有孤儿行！
        if time_anchors[0] > start_data_idx:
            block_starts.append((start_data_idx, "")) # 完美回收：插入表头底线索引
            
        for idx in time_anchors:
            block_starts.append((idx, ""))

    # ================= 🌟 孤儿行回收结束 =================

    if not block_starts:
        print(f"❌ [Worker] 致命错误：未能从横线区域提取到任何有效的时间(00:00 格式)！")
        if len(merged_y) > 1:
            cv2.imwrite(os.path.join(output_base, "debug_time_zone.png"), img[merged_y[-2]:merged_y[-1], :time_col_width])
        return

    vis_img = img.copy()
    success_count = 0
    
    for idx in range(len(block_starts)):
        try:
            start_node_idx = block_starts[idx][0]
            y_start = merged_y[start_node_idx]
            y_end = merged_y[block_starts[idx+1][0]] if idx < len(block_starts) - 1 else merged_y[-1]

            final_block = np.vstack((header, img[y_start:y_end, :]))
            header_h = header.shape[0]
            
            # 🌟 红线机制：在拼接好的块数据区画红色垂直辅助线（移植自王主任 cutter）
            for x_line in all_sub_lines:
                cv2.line(final_block, (x_line, header_h), (x_line, final_block.shape[0]), (0, 0, 255), 4)
            
            # 使用全局计算好的坐标切分（替代原来每块独立算的 split_into_three_columns）
            part_L = final_block[:, 0:global_x1]
            part_M = final_block[:, global_x1:global_x2]
            part_R = final_block[:, global_x2:W]
            
            # 🌟 OCR辅助文本导出 + 图像优化（替代原来的直接 cv2.imwrite）
            save_part_with_ocr(part_L, output_base, f"block_{idx:02d}_L", ocr)
            save_part_with_ocr(part_M, output_base, f"block_{idx:02d}_M", ocr)
            save_part_with_ocr(part_R, output_base, f"block_{idx:02d}_R", ocr)
            
            # 画上诊断线，方便预览
            cv2.rectangle(vis_img, (2, y_start), (W-2, y_end), (255, 0, 0), 2) 
            cv2.line(vis_img, (global_x1, y_start), (global_x1, y_end), (0, 0, 255), 3)      
            cv2.line(vis_img, (global_x2, y_start), (global_x2, y_end), (0, 255, 0), 3)  
            success_count += 1
        except Exception as e: 
            print(f"⚠️ [Worker] 切割第 {idx} 块时出错: {e}")
            continue

    cv2.imwrite(os.path.join(output_base, "_block_preview.png"), vis_img)
    print(f"✅ [Worker] 图片切割顺利完成！共切出 {success_count} 行记录碎片。")

if __name__ == "__main__":
    main()
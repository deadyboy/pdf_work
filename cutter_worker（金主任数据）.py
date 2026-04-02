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

def split_into_three_columns(img):
    h, w = img.shape[:2]
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, h // 40)))
        v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=2)
        contours, _ = cv2.findContours(v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        col_xs = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if ch > h * 0.3: col_xs.append(x + cw // 2)
                
        col_xs = sorted(list(set(col_xs)))
        if len(col_xs) > 0:
            merged_xs = [col_xs[0]]
            for v in col_xs[1:]:
                if v - merged_xs[-1] > 10: merged_xs.append(v)
            col_xs = merged_xs

        if len(col_xs) >= 32: 
            x_split1, x_split2 = col_xs[20], col_xs[31]  
        else:
            x_split1, x_split2 = int(w * 0.41), int(w * 0.72)
    except Exception as e:
        print(f"⚠️ 列切割计算出错: {e}")
        x_split1, x_split2 = int(w * 0.41), int(w * 0.72)

    x_split1 = max(0, min(x_split1, w-1))
    x_split2 = max(x_split1, min(x_split2, w-1))

    return img[:, 0:x_split1], img[:, x_split1:x_split2], img[:, x_split2:w], x_split1, x_split2

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

    header = img[0:header_bottom_y, :]
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
            print(f"⚠️ [Worker] 扫描第 {i} 行时间区域时出错: {e}")
            continue

    if not block_starts:
        print(f"❌ [Worker] 致命错误：未能从横线区域提取到任何有效的时间(00:00 格式)！")
        # 截取第一行的时间区域看看是不是OCR认不出来
        if len(merged_y) > 1:
            cv2.imwrite(os.path.join(output_base, "debug_time_zone.png"), img[merged_y[-2]:merged_y[-1], :time_col_width])
        return

    vis_img = img.copy()
    success_count = 0
    
    for idx in range(len(block_starts)):
        try:
            start_node_idx, t_str = block_starts[idx]
            y_start = merged_y[start_node_idx]
            y_end = merged_y[block_starts[idx+1][0]] if idx < len(block_starts) - 1 else merged_y[-1]

            final_block = np.vstack((header, img[y_start:y_end, :]))
            part_L, part_M, part_R, x1, x2 = split_into_three_columns(final_block)
            
            cv2.imwrite(os.path.join(output_base, f"block_{idx:02d}_L.png"), part_L)
            cv2.imwrite(os.path.join(output_base, f"block_{idx:02d}_M.png"), part_M)
            cv2.imwrite(os.path.join(output_base, f"block_{idx:02d}_R.png"), part_R)
            
            cv2.rectangle(vis_img, (2, y_start), (W-2, y_end), (255, 0, 0), 2) 
            cv2.line(vis_img, (x1, y_start), (x1, y_end), (0, 0, 255), 2)      
            cv2.line(vis_img, (x2, y_start), (x2, y_end), (0, 255, 0), 2)  
            success_count += 1
        except Exception as e: 
            print(f"⚠️ [Worker] 切割第 {idx} 块时出错: {e}")
            continue

    cv2.imwrite(os.path.join(output_base, "_block_preview.png"), vis_img)
    print(f"✅ [Worker] 图片切割顺利完成！共切出 {success_count} 行记录碎片。")

if __name__ == "__main__":
    main()
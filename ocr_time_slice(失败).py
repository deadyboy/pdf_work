import os, re, math
import cv2
import numpy as np
from paddleocr import PaddleOCR

TIME_RE = re.compile(r"\b([01]?\d|2[0-3])[:：]([0-5]\d)\b")  # 0:00-23:59

def norm_time(s: str):
    s = s.replace("：", ":")
    m = TIME_RE.search(s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    return f"{hh:02d}:{mm:02d}"

def merge_close(vals, tol=18):
    vals = sorted(vals)
    merged = []
    for v in vals:
        if not merged or abs(v - merged[-1]) > tol:
            merged.append(v)
    return merged

def slice_by_time(
    img_path: str,
    out_dir: str,
    left_ratio: float = 0.16,      # 左侧时间列宽度占比（模板可微调）
    top_ratio: float = 0.10,       # 去掉页眉
    bot_ratio: float = 0.88,       # 去掉红字说明
    rows_per_slice: int = 3,
    pad_y: int = 25,               # 每片上下留白，避免切到字
    ocr_lang: str = "ch",
):
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    H, W, _ = img.shape

    # 1) 粗裁：去掉页眉/页脚，只保留数据区（先粗暴，跑通后再精修）
    y_top = int(H * top_ratio)
    y_bot = int(H * bot_ratio)
    core = img[y_top:y_bot, :]

    # 2) 只裁“左侧时间列”做 OCR（减少干扰、提高命中率）
    x_right = int(W * left_ratio)
    time_strip = core[:, :x_right]

    # 3) 预处理：放大 + 灰度 + 轻二值化（提升时间识别）
    gray = cv2.cvtColor(time_strip, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 10
    )

    # 🔧 关键修复：把二值图“伪装回3通道”
    bw3 = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    # 4) PaddleOCR：只识别，不分类方向（可按需调整）
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
    result = ocr.ocr(bw3, cls=True)

    # 5) 收集所有识别到的时间戳及其 y 坐标
    hits = []
    for line in result[0]:
        box, (text, score) = line
        t = norm_time(text)
        if not t:
            continue
        # box 是 4 点坐标，取中心 y
        ys = [p[1] for p in box]
        y_center = float(sum(ys) / 4.0)

        # 注意：我们对 time_strip 做了 1.8 倍缩放，所以要除回去
        y_center = y_center / 1.8

        hits.append((y_center, t, float(score), text))

    # 按 y 排序
    hits.sort(key=lambda x: x[0])

    # debug 输出
    debug_txt = os.path.join(out_dir, "time_hits.txt")
    with open(debug_txt, "w", encoding="utf-8") as f:
        for y, t, sc, raw in hits:
            f.write(f"y={y:.1f}\t{t}\tscore={sc:.2f}\traw={raw}\n")

    if len(hits) < 4:
        print("⚠️ 时间戳命中太少，可能需要调 left_ratio/top_ratio/bot_ratio 或增强预处理。")
        print("hits saved to:", debug_txt)

    # 6) 取 y 坐标做聚类合并（同一行可能识别出多个相近框）
    ys = merge_close([int(round(y)) for y, *_ in hits], tol=22)

    # 7) 生成“行边界”：用相邻时间 y 的中点作为分割线
    #    行区间大致为 [mid(prev,next), mid(next,nextnext)] 的形式
    #    更简单：直接用时间 y 作为每行“锚点”，切片时向上下扩展
    #    这里采用：按时间锚点切成连续块（更稳）
    core_h = core.shape[0]
    # 给锚点补边界
    if ys and ys[0] > 40:
        ys = [0] + ys
    if ys and (core_h - ys[-1] > 40):
        ys = ys + [core_h]

    # 相邻锚点之间作为“行带”
    bands = list(zip(ys[:-1], ys[1:]))

    # 8) 按 rows_per_slice 合并行带，切出全宽切片
    slice_paths = []
    k = 0
    for i in range(0, len(bands), rows_per_slice):
        chunk = bands[i:i+rows_per_slice]
        if not chunk:
            continue
        y1 = max(0, chunk[0][0] - pad_y)
        y2 = min(core_h, chunk[-1][1] + pad_y)
        crop = core[y1:y2, :]  # 全宽（后面可再列分区）

        k += 1
        out_path = os.path.join(out_dir, f"slice_{k:02d}.png")
        cv2.imwrite(out_path, crop)
        slice_paths.append(out_path)

    # 9) 画 debug 图：把识别到的时间 y 画在原图上，便于你肉眼检查
    dbg = core.copy()
    for y in ys:
        cv2.line(dbg, (0, y), (dbg.shape[1], y), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(out_dir, "debug_lines.png"), dbg)

    print("✅ time hits:", len(hits), "anchors:", len(ys), "slices:", len(slice_paths))
    print("outputs in:", out_dir)
    return slice_paths, debug_txt

if __name__ == "__main__":
    img_path = "/data1/jianf/left_side_medical_record.jpg"
    out_dir = "/data1/jianf/新提取pdf/data/time_ocr_slices"
    slice_by_time(img_path, out_dir)

"""纸张对折打孔展开预览 —— Flask 后端。

几何模型
========
- 原纸坐标系：原点 (0,0) 在左下角，x 向右、y 向上，单位 mm。
- 每次对折沿当前折叠面的水平或垂直中线，当前面对应尺寸减半。
- 折叠后逐条跟踪当前面四条边的身份：
  * "cut"  —— 开口边（对应原纸边界）
  * "fold" —— 折痕边
  重要不变量：历史上所有折痕最终都落在当前折叠面的边界上。因此打孔点只要
  与当前面四条边界的距离都大于打孔半径（带容差），展开后所有孔就不会压在
  原纸边界或任何折痕上，结果唯一、不含糊。
- 展开：按折叠顺序逆推（物理 mm 坐标）。每展开一层，静止层恒等映射，
  翻折层关于该层中线镜像；最终得到原纸上的全部孔位并按坐标去重。
- 反查打孔：把原纸上的目标孔心按折叠顺序正推（静止半恒等、翻折半关于
  中线镜像），逐层折回最终折叠面，得到归一化孔心；再用同一展开算法列出
  这一次打孔必然生成的全部孔位。目标孔心距纸张边界/折痕不大于打孔半径时
  拒绝并说明具体原因。
- 全部折痕展开后构成 (2^nv - 1) x (2^nh - 1) 的等分网格。
- 原纸保护区：原纸上至多 MAX_ZONES 个轴对齐矩形（mm 坐标），绘入展开图。
  任一展开孔圆与任一保护区相交或相切即违规：指出原打孔点、触碰的展开孔
  及保护区，不生成展开 SVG、禁止下载；反查打孔的待确认结果同样判定。
  保护区必须完整位于原纸内，纸张尺寸改动导致越界时提示修正。
"""

from __future__ import annotations

import math
import os

from flask import Flask, jsonify, render_template, request, Response

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 配置（可用环境变量覆盖）
# ---------------------------------------------------------------------------
MAX_FOLDS = int(os.environ.get("MAX_FOLDS", "8"))
MAX_HOLES = int(os.environ.get("MAX_HOLES", "24"))
MAX_ZONES = int(os.environ.get("MAX_ZONES", "8"))
MIN_DIM = 20.0          # mm
MAX_DIM = 2000.0        # mm
MIN_RADIUS = 0.5        # mm
MAX_RADIUS = 50.0       # mm
EDGE_TOL = 0.05         # mm，边界判定容差（浮点误差 + 必须严格在内侧）

ALLOWED_FOLDS = ("vL", "vR", "hT", "hB")
EDGE_NAMES = {
    "fold": {"x0": "左侧折痕边", "x1": "右侧折痕边",
             "y0": "下侧折痕边", "y1": "上侧折痕边"},
    "cut": {"x0": "纸张左边界", "x1": "纸张右边界",
            "y0": "纸张下边界", "y1": "纸张上边界"},
}


# ---------------------------------------------------------------------------
# 折叠模型
# ---------------------------------------------------------------------------
def folded_state(width, height, folds):
    """按折叠顺序推进，返回 (当前面宽, 当前面高, 四条边身份, 垂直/水平次数)。"""
    w, h = float(width), float(height)
    # 初始四条边都是原纸开口边界
    edges = {"x0": "cut", "x1": "cut", "y0": "cut", "y1": "cut"}
    nv = nh = 0
    for f in folds:
        if f in ("vL", "vR"):
            w /= 2.0
            nv += 1
            # 垂直对折：折痕落在当前面中线上。vR 右半折向左，折痕边是当前面
            # 右边（x1）；vL 左半折向右，折痕边是当前面左边（x0）。
            edges["x1" if f == "vR" else "x0"] = "fold"
        else:
            h /= 2.0
            nh += 1
            # 水平对折：hT 上半折向下，折痕边是当前面上边（y1）；
            # hB 下半折向上，折痕边是当前面下边（y0）。
            edges["y0" if f == "hB" else "y1"] = "fold"
    return w, h, edges, nv, nh


def unfold_points(points_mm, folds, width, height):
    """逆推孔位。输入最终折叠面局部 mm 坐标（原点为该面左下角），
    返回原纸 mm 坐标列表。

    逆推必须使用物理坐标：每展开一层，面宽/面高翻倍，中线位置随之变化，
    归一化坐标无法表达这一点。
    """
    cur = [(float(p["x"]), float(p["y"])) for p in points_mm]
    cw = width / (2.0 ** sum(1 for f in folds if f in ("vL", "vR")))
    ch = height / (2.0 ** sum(1 for f in folds if f in ("hT", "hB")))
    for f in reversed(folds):
        nxt = []
        if f in ("vL", "vR"):
            pw = 2.0 * cw          # 展开前一面的宽度
            line = pw / 2.0
            if f == "vR":          # 右侧折向左：静止左半恒等，右半镜像
                for x, y in cur:
                    nxt.append((x, y))
                    nxt.append((2.0 * line - x, y))
            else:                  # vL 左侧折向右：静止右半，左半镜像
                for x, y in cur:
                    nxt.append((line + x, y))
                    nxt.append((line - x, y))
            cw = pw
        else:
            ph = 2.0 * ch
            line = ph / 2.0
            if f == "hT":          # 上侧折向下：静止下半恒等，上半镜像
                for x, y in cur:
                    nxt.append((x, y))
                    nxt.append((x, 2.0 * line - y))
            else:                  # hB 下侧折向上：静止上半，下半镜像
                for x, y in cur:
                    nxt.append((x, line + y))
                    nxt.append((x, line - y))
            ch = ph
        cur = nxt
    return cur


def fold_point_to_face(tx, ty, folds, width, height):
    """正推：把原纸 mm 坐标的目标点按折叠顺序逐层折回最终折叠面。

    与 unfold_points 互为逆映射：每折一次，静止半不变，翻折半关于该层
    中线镜像，当前面对应尺寸减半。返回折叠面局部 mm 坐标。
    """
    x, y = float(tx), float(ty)
    cw, ch = float(width), float(height)
    for f in folds:
        if f in ("vL", "vR"):
            line = cw / 2.0
            # vR 右半折向左：x -> line - |x - line|；vL 左半折向右：x -> |x - line|
            x = line - abs(x - line) if f == "vR" else abs(x - line)
            cw = line
        else:
            line = ch / 2.0
            # hT 上半折向下：y -> line - |y - line|；hB 下半折向上：y -> |y - line|
            y = line - abs(y - line) if f == "hT" else abs(y - line)
            ch = line
    return x, y


def dedupe_points(points, tol=1e-5):
    """按物理 mm 坐标去重（重合位置只保留一个）。"""
    out = []
    for x, y in points:
        if any(abs(x - q[0]) <= tol and abs(y - q[1]) <= tol for q in out):
            continue
        out.append((x, y))
    return [{"x": round(x, 6), "y": round(y, 6)} for x, y in out]


def check_point(px, py, w, h, radius, edges):
    """校验单个打孔点，返回中文错误信息；合法返回 None。"""
    r = radius + EDGE_TOL
    checks = (("x0", px), ("x1", w - px), ("y0", py), ("y1", h - py))
    for key, dist in checks:
        if dist < r:
            kind = edges[key]
            return (f"孔位 ({px:.2f}, {py:.2f}) 距{EDGE_NAMES[kind][key]}仅 "
                    f"{max(dist, 0.0):.2f}mm，小于打孔半径 {radius:g}mm"
                    f"（{'折痕' if kind == 'fold' else '纸张边界'}，"
                    f"孔必须完整落在纸面内侧），请调整孔位。")
    return None


def check_target_on_paper(tx, ty, width, height, radius, nv, nh):
    """校验原纸上的目标孔心：打孔半径不能触及纸张边界或任何折痕。

    与折叠面上的 check_point 几何上等价（折叠就是把边界/折痕网格
    折到当前面四边），这里从原纸视角给出更直观的中文原因。
    """
    r = radius + EDGE_TOL
    bounds = (("纸张左边界", tx), ("纸张右边界", width - tx),
              ("纸张下边界", ty), ("纸张上边界", height - ty))
    for name, dist in bounds:
        if dist < r:
            return (f"目标孔心 ({tx:.2f}, {ty:.2f}) 距{name}仅 "
                    f"{max(dist, 0.0):.2f}mm，小于打孔半径 {radius:g}mm"
                    "（孔会超出纸张边界），请把目标点移离边界。")
    for k in range(1, 2 ** nv):
        pos = width * k / 2 ** nv
        dist = abs(tx - pos)
        if dist < r:
            return (f"目标孔心 ({tx:.2f}, {ty:.2f}) 距垂直折痕 x={pos:.2f}mm 仅 "
                    f"{dist:.2f}mm，小于打孔半径 {radius:g}mm"
                    "（孔会压在折痕上），请把目标点移离折痕。")
    for k in range(1, 2 ** nh):
        pos = height * k / 2 ** nh
        dist = abs(ty - pos)
        if dist < r:
            return (f"目标孔心 ({tx:.2f}, {ty:.2f}) 距水平折痕 y={pos:.2f}mm 仅 "
                    f"{dist:.2f}mm，小于打孔半径 {radius:g}mm"
                    "（孔会压在折痕上），请把目标点移离折痕。")
    return None


def crease_grid(width, height, nv, nh):
    """计算展开后全部折痕线段（mm，原纸坐标，y 自下而上）。"""
    lines = []
    for k in range(1, 2 ** nv):
        lines.append({"orient": "v", "pos": round(width * k / 2 ** nv, 6)})
    for k in range(1, 2 ** nh):
        lines.append({"orient": "h", "pos": round(height * k / 2 ** nh, 6)})
    return lines


# ---------------------------------------------------------------------------
# 原纸保护区（轴对齐矩形，原纸 mm 坐标）
# ---------------------------------------------------------------------------
def circle_touches_zone(cx, cy, radius, z):
    """孔圆与轴对齐矩形保护区是否相交或相切（含 EDGE_TOL 容差）。

    圆心到矩形的最近距离 <= 半径即相交/相切；保护区为禁入区域，
    相切也算违规。
    """
    dx = max(z["x"] - cx, 0.0, cx - (z["x"] + z["w"]))
    dy = max(z["y"] - cy, 0.0, cy - (z["y"] + z["h"]))
    limit = radius + EDGE_TOL
    return dx * dx + dy * dy <= limit * limit


def zone_out_of_bounds(zones, width, height):
    """返回第一个越出原纸的保护区下标；都在纸内返回 None。"""
    for i, z in enumerate(zones):
        if (z["x"] + z["w"] > width + EDGE_TOL
                or z["y"] + z["h"] > height + EDGE_TOL):
            return i
    return None


def zone_bounds_msg(i, z, width, height):
    return (f"保护区 #{i + 1}（x {z['x']:g} ~ {z['x'] + z['w']:g}mm，"
            f"y {z['y']:g} ~ {z['y'] + z['h']:g}mm）超出原纸 "
            f"{width:g} × {height:g}mm 的范围，请修正或删除该保护区。")


def zone_hit_msg(prefix, hx, hy, zi, z, radius):
    return (f"{prefix}展开后的孔 ({hx:.2f}, {hy:.2f})mm 与保护区 #{zi + 1}"
            f"（x {z['x']:g} ~ {z['x'] + z['w']:g}mm，"
            f"y {z['y']:g} ~ {z['y'] + z['h']:g}mm）相交或相切"
            f"（孔半径 {radius:g}mm，保护区为禁入区域），"
            "请移开打孔点或调整保护区。")


def find_zone_hit(raw_holes, zones, radius, n_images):
    """在展开后的原始孔位（未去重）中找第一个触碰保护区的孔。

    raw_holes 按 unfold_points 的输出顺序排列：第 k 个孔来自第
    k // n_images 个打孔点（每个打孔点恰好展开出 n_images = 2^折数 个
    原始孔位）。返回 (打孔点下标, 孔坐标, 保护区下标) 或 None。
    """
    if not zones or not raw_holes:
        return None
    for k, (x, y) in enumerate(raw_holes):
        src = k // n_images
        for zi, z in enumerate(zones):
            if circle_touches_zone(x, y, radius, z):
                return src, {"x": round(x, 6), "y": round(y, 6)}, zi
    return None


# ---------------------------------------------------------------------------
# SVG 生成
# ---------------------------------------------------------------------------
def _svg_header(w_mm, h_mm):
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w_mm:.3f}mm" height="{h_mm:.3f}mm" '
        f'viewBox="0 0 {w_mm:.3f} {h_mm:.3f}">\n'
        f'<rect x="0" y="0" width="{w_mm:.3f}" height="{h_mm:.3f}" '
        'fill="#ffffff" stroke="#1f2937" stroke-width="0.35"/>\n'
    )


def _circle(cx, cy, r, extra=""):
    return (f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}" '
            f'fill="#dc2626" stroke="#7f1d1d" stroke-width="0.12"{extra}/>\n')


def build_unfolded_svg(width, height, holes, creases, radius, nv, nh, zones):
    """展开图 SVG。y 由数学坐标（自下而上）换算为 SVG 坐标（自上而下）。"""
    w, h, r = float(width), float(height), float(radius)
    parts = [_svg_header(w, h)]
    parts.append(
        f'<desc>纸张 {w:g}x{h:g}mm；垂直对折 {nv} 次、水平对折 {nh} 次；'
        f'保护区 {len(zones)} 个；共 {len(holes)} 个孔</desc>\n')
    for c in creases:
        if c["orient"] == "v":
            parts.append(
                f'<line x1="{c["pos"]:.3f}" y1="0" x2="{c["pos"]:.3f}" y2="{h:.3f}" '
                'stroke="#9ca3af" stroke-width="0.2" stroke-dasharray="2 1.5"/>\n')
        else:
            yy = h - c["pos"]
            parts.append(
                f'<line x1="0" y1="{yy:.3f}" x2="{w:.3f}" y2="{yy:.3f}" '
                'stroke="#9ca3af" stroke-width="0.2" stroke-dasharray="2 1.5"/>\n')
    # 原纸保护区：绿色半透明填充 + 虚线边框，标序号
    for zi, z in enumerate(zones):
        zy_svg = h - (z["y"] + z["h"])
        parts.append(
            f'<rect x="{z["x"]:.3f}" y="{zy_svg:.3f}" '
            f'width="{z["w"]:.3f}" height="{z["h"]:.3f}" '
            'fill="#16a34a" fill-opacity="0.10" stroke="#15803d" '
            'stroke-width="0.4" stroke-dasharray="2.5 1.8"/>\n')
        if z["w"] >= 8.0 and z["h"] >= 4.5:
            parts.append(
                f'<text x="{z["x"] + 1.0:.3f}" y="{zy_svg + 3.4:.3f}" '
                f'font-size="3" fill="#15803d">保{zi + 1}</text>\n')
    for hp in holes:
        parts.append(_circle(hp["x"], h - hp["y"], r))
    parts.append("</svg>\n")
    return "".join(parts)


def build_folded_svg(w, h, edges, points, radius, error_idx):
    """折叠面 SVG，使用 SVG 显示坐标（y 自上而下）。

    纸面在 viewBox 中内缩 pad；客户端靠返回的 pad / vbWidth / vbHeight
    把点击位置换算回折叠面 mm 坐标。
    """
    pad = max(radius * 2.2, 3.0)
    vb_w, vb_h = w + 2 * pad, h + 2 * pad

    def line(x1, y1, x2, y2, fold):
        color, dash, sw = (("#2563eb", "6 4", 0.9) if fold
                           else ("#1f2937", "none", 1.2))
        da = f' stroke-dasharray="{dash}"' if fold else ""
        return (f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
                f'stroke="{color}" stroke-width="{sw}"{da} stroke-linecap="round"/>\n')

    y_top, y_bot = pad, pad + h
    parts = [
        f'<svg id="folded-svg" viewBox="0 0 {vb_w:.3f} {vb_h:.3f}" '
        'preserveAspectRatio="xMidYMid meet" role="img" '
        'aria-label="折叠面，点击打孔">\n',
        # 点击层（覆盖整个 viewBox）
        f'<rect x="0" y="0" width="{vb_w:.3f}" height="{vb_h:.3f}" fill="transparent"/>\n',
        f'<rect x="{pad:.3f}" y="{pad:.3f}" width="{w:.3f}" height="{h:.3f}" '
        'fill="#fdf6e3" stroke="none"/>\n',
        line(pad, y_bot, pad + w, y_bot, edges["y0"] == "fold"),   # 下边
        line(pad, y_top, pad + w, y_top, edges["y1"] == "fold"),   # 上边
        line(pad, y_top, pad, y_bot, edges["x0"] == "fold"),       # 左边
        line(pad + w, y_top, pad + w, y_bot, edges["x1"] == "fold"),  # 右边
        # 安全区域：孔心进入此绿虚线框内才不会碰边/折痕
        f'<rect x="{pad + radius:.3f}" y="{pad + radius:.3f}" '
        f'width="{max(w - 2 * radius, 0):.3f}" height="{max(h - 2 * radius, 0):.3f}" '
        'fill="none" stroke="#16a34a" stroke-width="0.3" '
        'stroke-dasharray="1.5 1.5" opacity="0.55"/>\n',
    ]
    for i, p in enumerate(points):
        extra = ' data-hole="1"'
        if i == error_idx:
            extra += ' fill="#f59e0b"'
        # 数学坐标 y（自下而上）-> 显示坐标（自上而下）
        parts.append(_circle(pad + p["x"], pad + h - p["y"], radius, extra))
    parts.append("</svg>\n")
    return "".join(parts), pad, vb_w, vb_h


# ---------------------------------------------------------------------------
# 请求处理
# ---------------------------------------------------------------------------
def _bad(msg, code=400):
    return jsonify({"ok": False, "error": msg}), code


def _num(obj, key, lo, hi, name):
    val = obj.get(key)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name}必须是数字")
    val = float(val)
    if not math.isfinite(val) or val < lo or val > hi:
        raise ValueError(f"{name}须在 {lo:g} ~ {hi:g} 之间")
    return val


def _validate_sheet(payload):
    """校验纸张尺寸、打孔半径与折法，返回 (width, height, radius, folds)。"""
    width = _num(payload, "width", MIN_DIM, MAX_DIM, "纸张宽度")
    height = _num(payload, "height", MIN_DIM, MAX_DIM, "纸张高度")
    radius = _num(payload, "radius", MIN_RADIUS, MAX_RADIUS, "打孔半径")

    folds = payload.get("folds", [])
    if not isinstance(folds, list) or len(folds) > MAX_FOLDS:
        raise ValueError(f"折叠次数须为 0 ~ {MAX_FOLDS} 次")
    for f in folds:
        if f not in ALLOWED_FOLDS:
            raise ValueError("存在非法折叠方向，可选：vL / vR / hT / hB")
    return width, height, radius, folds


def _check_face_capacity(w, h, radius):
    """折叠面必须放得下打孔器（直径 + 容差）。"""
    if w < 2 * radius + 2 * EDGE_TOL or h < 2 * radius + 2 * EDGE_TOL:
        raise ValueError(
            f"折叠后纸面为 {w:g} × {h:g}mm，放不下直径 {2 * radius:g}mm 的打孔器，"
            "请减少折叠次数或减小打孔半径。")


def _validate_zones(payload):
    """校验保护区结构：至多 MAX_ZONES 个轴对齐矩形 {x, y, w, h}（原纸 mm）。

    只校验格式与取值范围；是否越出原纸由调用方结合纸张尺寸判定
    （compute 按孔位错误同款方式返回 200 + error，reverse 抛 ValueError）。
    """
    zones = payload.get("zones", [])
    if not isinstance(zones, list) or len(zones) > MAX_ZONES:
        raise ValueError(f"保护区数量须为 0 ~ {MAX_ZONES} 个")
    out = []
    for i, z in enumerate(zones):
        if not isinstance(z, dict):
            raise ValueError("保护区格式错误")
        zx = _num(z, "x", 0.0, MAX_DIM, f"保护区 #{i + 1} 的 x")
        zy = _num(z, "y", 0.0, MAX_DIM, f"保护区 #{i + 1} 的 y")
        zw = _num(z, "w", 0.0, MAX_DIM, f"保护区 #{i + 1} 的宽")
        zh = _num(z, "h", 0.0, MAX_DIM, f"保护区 #{i + 1} 的高")
        if zw <= 0.0 or zh <= 0.0:
            raise ValueError(f"保护区 #{i + 1} 的宽和高必须大于 0")
        out.append({"x": zx, "y": zy, "w": zw, "h": zh})
    return out


def compute(payload):
    """核心计算。成功返回结果 dict，失败抛出 ValueError。"""
    if not isinstance(payload, dict):
        raise ValueError("请求体必须是 JSON 对象")

    width, height, radius, folds = _validate_sheet(payload)
    zones = _validate_zones(payload)

    points = payload.get("points", [])
    if not isinstance(points, list) or len(points) > MAX_HOLES:
        raise ValueError(f"打孔点数量须为 0 ~ {MAX_HOLES} 个")
    norm_pts = []
    for p in points:
        if not isinstance(p, dict):
            raise ValueError("打孔点格式错误")
        norm_pts.append({"x": _num(p, "x", 0.0, 1.0, "打孔点 x"),
                         "y": _num(p, "y", 0.0, 1.0, "打孔点 y")})

    w, h, edges, nv, nh = folded_state(width, height, folds)
    _check_face_capacity(w, h, radius)

    # 客户端坐标是归一化的；换算为折叠面 mm 坐标
    mm_pts = [{"x": p["x"] * w, "y": p["y"] * h} for p in norm_pts]

    error_msg, error_idx = None, -1
    zone_error = None
    holes, creases, unfolded_svg = [], [], None

    # 保护区必须完整位于原纸内（纸张尺寸改小导致越界时在此提示修正）
    oob = zone_out_of_bounds(zones, width, height)
    if oob is not None:
        error_msg = zone_bounds_msg(oob, zones[oob], width, height)
        zone_error = {"kind": "bounds", "zoneIndex": oob}

    if error_msg is None:
        for i, p in enumerate(mm_pts):
            err = check_point(p["x"], p["y"], w, h, radius, edges)
            if err:
                error_msg, error_idx = err, i
                break

    if error_msg is None:
        raw = unfold_points(mm_pts, folds, width, height)
        # 任一展开孔圆与任一保护区相交/相切：指出原打孔点、展开孔与保护区
        hit = find_zone_hit(raw, zones, radius, 2 ** len(folds))
        if hit:
            src, hole, zi = hit
            p = mm_pts[src]
            error_msg, error_idx = zone_hit_msg(
                f"打孔点 #{src + 1}（折叠面 ({p['x']:.2f}, {p['y']:.2f})mm）",
                hole["x"], hole["y"], zi, zones[zi], radius), src
            zone_error = {"kind": "hit", "pointIndex": src,
                          "hole": hole, "zoneIndex": zi}
        else:
            holes = dedupe_points(raw)
            creases = crease_grid(width, height, nv, nh)
            unfolded_svg = build_unfolded_svg(
                width, height, holes, creases, radius, nv, nh, zones)

    folded_svg, pad, vb_w, vb_h = build_folded_svg(
        w, h, edges, mm_pts, radius, error_idx)

    return {
        "ok": True,
        "folded": {
            "width": round(w, 6), "height": round(h, 6),
            "layers": 2 ** len(folds),
            "edges": edges,
            "pad": round(pad, 6),
            "vbWidth": round(vb_w, 6), "vbHeight": round(vb_h, 6),
            "svg": folded_svg,
        },
        "creases": creases,
        "holeCount": len(holes),
        "expectedCount": len(mm_pts) * (2 ** len(folds)),
        "unfoldedSvg": unfolded_svg,
        "error": error_msg,
        "errorIndex": error_idx,
        "zoneError": zone_error,
    }


def reverse_lookup(payload):
    """反查打孔：原纸上的目标孔心 -> 最终折叠面的归一化孔心 + 全部连带孔位。

    目标点或折回后的孔心若使打孔半径触及纸张边界/折痕，抛出带具体原因的
    ValueError，客户端据此禁止确认。连带孔与保护区相交/相切时同样拒绝。
    """
    if not isinstance(payload, dict):
        raise ValueError("请求体必须是 JSON 对象")

    width, height, radius, folds = _validate_sheet(payload)
    zones = _validate_zones(payload)

    target = payload.get("target")
    if not isinstance(target, dict):
        raise ValueError("缺少目标孔心 target（原纸 mm 坐标）")
    tx = _num(target, "x", 0.0, width, "目标孔心 x")
    ty = _num(target, "y", 0.0, height, "目标孔心 y")

    w, h, edges, nv, nh = folded_state(width, height, folds)
    _check_face_capacity(w, h, radius)

    # 保护区必须完整位于原纸内
    oob = zone_out_of_bounds(zones, width, height)
    if oob is not None:
        raise ValueError(zone_bounds_msg(oob, zones[oob], width, height))

    # 目标孔心在原纸上的校验（边界 + 全部折痕）
    err = check_target_on_paper(tx, ty, width, height, radius, nv, nh)
    if err:
        raise ValueError(err)

    # 按当前折叠顺序及折向逐层折回最终折叠面
    fx, fy = fold_point_to_face(tx, ty, folds, width, height)

    # 反查孔心在折叠面上的校验（与上面的校验几何等价，作浮点兜底）
    err = check_point(fx, fy, w, h, radius, edges)
    if err:
        raise ValueError(f"反查到折叠面的{err}")

    # 复用现有展开算法，列出这一次打孔必然生成的全部孔位
    raw = unfold_points([{"x": fx, "y": fy}], folds, width, height)

    # 与折叠面打孔同一判定：任一连带孔圆触碰保护区即拒绝
    hit = find_zone_hit(raw, zones, radius, 2 ** len(folds))
    if hit:
        _, hole, zi = hit
        raise ValueError(zone_hit_msg(
            f"反查打孔（目标孔心 ({tx:.2f}, {ty:.2f})mm）",
            hole["x"], hole["y"], zi, zones[zi], radius))

    holes = dedupe_points(raw)

    return {
        "ok": True,
        "target": {"x": round(tx, 6), "y": round(ty, 6)},
        "center": {"x": round(fx / w, 6), "y": round(fy / h, 6)},
        "centerMm": {"x": round(fx, 6), "y": round(fy, 6)},
        "foldedWidth": round(w, 6),
        "foldedHeight": round(h, 6),
        "holes": holes,
        "holeCount": len(holes),
    }


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        max_folds=MAX_FOLDS, max_holes=MAX_HOLES, max_zones=MAX_ZONES,
        min_dim=MIN_DIM, max_dim=MAX_DIM,
        min_radius=MIN_RADIUS, max_radius=MAX_RADIUS)


@app.route("/api/unfold", methods=["POST"])
def api_unfold():
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return _bad("请求不是合法 JSON")
    try:
        return jsonify(compute(payload))
    except ValueError as exc:
        return _bad(str(exc))


@app.route("/api/reverse", methods=["POST"])
def api_reverse():
    """反查打孔：入参 {width, height, radius, folds, zones, target:{x,y}(原纸 mm)}。"""
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return _bad("请求不是合法 JSON")
    try:
        return jsonify(reverse_lookup(payload))
    except ValueError as exc:
        return _bad(str(exc))


@app.route("/api/download", methods=["POST"])
def api_download():
    """只在计算合法时返回可下载的展开 SVG 文件。"""
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return _bad("请求不是合法 JSON")
    try:
        result = compute(payload)
    except ValueError as exc:
        return _bad(str(exc))
    if result["error"] or not result["unfoldedSvg"]:
        return _bad(result["error"] or "存在不合法孔位，无法生成展开图")
    return Response(
        result["unfoldedSvg"],
        mimetype="image/svg+xml",
        headers={"Content-Disposition":
                 'attachment; filename="unfolded.svg"'})


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)

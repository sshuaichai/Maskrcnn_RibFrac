from PIL import Image as PILImage, ImageDraw, ImageFont, ImageColor
import numpy as np

# 已去除较暗的颜色
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan', 'DeepPink',
    'DeepSkyBlue', 'DodgerBlue', 'FloralWhite', 'Fuchsia', 'Gainsboro',
    'GhostWhite', 'Gold', 'GoldenRod', 'Salmon', 'Tan', 'HoneyDew', 'HotPink',
    'IndianRed', 'Ivory', 'Khaki', 'Lavender', 'LavenderBlush', 'LawnGreen',
    'LemonChiffon', 'LightBlue', 'LightCoral', 'LightCyan', 'LightGoldenRodYellow',
    'LightGray', 'LightGrey', 'LightGreen', 'LightPink', 'LightSalmon',
    'LightSeaGreen', 'LightSkyBlue', 'LightSlateGray', 'LightSlateGrey',
    'LightSteelBlue', 'LightYellow', 'Lime', 'LimeGreen', 'Linen', 'Magenta',
    'MediumAquaMarine', 'MediumOrchid', 'MediumPurple', 'MediumSeaGreen',
    'MediumSlateBlue', 'MediumSpringGreen', 'MediumTurquoise', 'MediumVioletRed',
    'MintCream', 'MistyRose', 'Moccasin', 'NavajoWhite', 'OldLace', 'Orange',
    'OrangeRed', 'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise',
    'PaleVioletRed', 'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum',
    'PowderBlue', 'Red', 'RosyBrown', 'RoyalBlue', 'SandyBrown', 'SeaShell',
    'Silver', 'SkyBlue', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White', 'WhiteSmoke',
    'Yellow', 'YellowGreen'
]

def draw_text(image, draw, box, cls, score, category_index, color, font='arial.ttf', font_size=24, text_list=[]):
    """
    将目标边界框和类别信息绘制到图片上
    每次绘制文本框时，会先计算初始位置，然后检查这个位置是否与已有的文本框重叠。
    如果重叠，则调整位置，直到找到一个不重叠的位置为止。最终，绘制文本框和文本，并记录这个文本框的位置用于后续检查。
    """
    try:
        font = ImageFont.truetype(font, font_size)
    except IOError:
        font = ImageFont.load_default()

    left, top, right, bottom = box
    display_str = f"{category_index[str(cls)]}: {int(100 * score)}%"
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)

    # 初始文本框位置，放在边界框的上方，如果上方空间不够则放在边界框的下方
    text_top = top - text_height - 2 * margin
    if text_top < 0:
        text_top = bottom + 2 * margin
    text_bottom = text_top + text_height + 2 * margin

    # 遍历 text_list，检查当前文本框位置是否与已有的文本框重叠。
    # 如果重叠，则将文本框向上或向下移动，直到找到一个不重叠的位置。如果移动到顶部边缘之外，则将文本框放在边界框的下方。
    for previous_left, previous_top, previous_right, previous_bottom in text_list:
        if not (left > previous_right or right < previous_left or text_bottom < previous_top or text_top > previous_bottom):
            text_top -= (text_height + 2 * margin)
            text_bottom = text_top + text_height + 2 * margin
            if text_top < 0:
                text_top = bottom + 2 * margin
                text_bottom = text_top + text_height + 2 * margin

    # 在调整好位置后，绘制文本和透明背景。
    # Create a transparent background
    text_background = PILImage.new('RGBA', (text_width, text_height), (0, 0, 0, 0))
    draw_text_background = ImageDraw.Draw(text_background)

    # Draw text onto the transparent background
    draw_text_background.text((0, 0), display_str, fill=color, font=font)

    # Paste the transparent background onto the image
    image.paste(text_background, (int(left + margin), int(text_top)), text_background)

    # 将当前文本框的位置记录到 text_list 中，以便后续文本框绘制时进行重叠检查。
    text_list.append((left, text_top, left + text_width, text_top + text_height + 2 * margin))

def draw_masks(image, masks, colors, thresh=0.7, alpha=0.5):
    np_image = np.array(image)
    masks = np.where(masks > thresh, True, False)

    img_to_draw = np.copy(np_image)
    for mask, color in zip(masks, colors):
        img_to_draw[mask] = color

    out = np_image * (1 - alpha) + img_to_draw * alpha
    return PILImage.fromarray(out.astype(np.uint8))

def draw_objs(image: PILImage,
              boxes: np.ndarray = None,
              classes: np.ndarray = None,
              scores: np.ndarray = None,
              masks: np.ndarray = None,
              category_index: dict = None,
              box_thresh: float = 0.1,
              mask_thresh: float = 0.5,
              line_thickness: int = 8,
              font: str = 'arial.ttf',
              font_size: int = 24,
              draw_boxes_on_image: bool = True,
              draw_masks_on_image: bool = True):
    idxs = np.greater(scores, box_thresh)
    boxes = boxes[idxs]
    classes = classes[idxs]
    scores = scores[idxs]
    if masks is not None and len(masks.shape) == 3 and masks.shape[0] == len(idxs):
        masks = masks[idxs]
    else:
        masks = None

    if len(boxes) == 0:
        return image

    colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in classes]

    text_list = []  # Initialize list to keep track of text positions

    if draw_boxes_on_image:
        draw = ImageDraw.Draw(image)
        for box, cls, score, color in zip(boxes, classes, scores, colors):
            left, top, right, bottom = box
            draw.line(
                [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
                width=line_thickness,
                fill=color
            )
            draw_text(image, draw, box.tolist(), int(cls), float(score), category_index, color, font, font_size, text_list)

    if draw_masks_on_image and (masks is not None):
        image = draw_masks(image, masks, colors, mask_thresh)

    return image

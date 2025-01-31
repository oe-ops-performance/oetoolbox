from openpyxl.styles import PatternFill, Border, Side, Alignment, Font

colors = {
    "blk": "000000",  # black
    "wht": "FFFFFF",  # white
    "grey1": "F2F2F2",  # light grey
    "grey1a": "E6E6E6",  # lt-med-lt grey
    "grey2": "D9D9D9",  # med-light grey
    "grey3": "BFBFBF",  # medium grey
    "grey4": "A6A6A6",  # med-dark grey
    "grey5": "808080",  # dark grey
    "rust1": "FFDFC5",  # light rust/orange
    "rust2": "D55E00",  # dark rust/orange
    "blue1": "C1E4F7",  # light blue
    "blue2": "1777AD",  # dark blue
    "red": "FF0000",  # red
    "orng": "F79646",  # orange
    "yllw": "FFFF00",  # yellow
    "ylgr": "E2FF33",  # yellow-green
    "bluuu": "305496",
}

"""fill"""
# formatting functions for openpyxl file generation scripts
fill_dict = {id_: PatternFill("solid", fgColor=color) for id_, color in colors.items()}

"""border"""
s = {  # sides
    "thin": Side(border_style="thin", color=colors["blk"]),
    "thin1": Side(border_style="thin", color=colors["grey3"]),
    "thin2": Side(border_style="thin", color=colors["grey4"]),
    "med": Side(border_style="medium", color=colors["blk"]),
    "med1": Side(border_style="medium", color=colors["grey3"]),
    "med2": Side(border_style="medium", color=colors["grey4"]),
    "doub": Side(border_style="double", color=colors["blk"]),
    "doub2": Side(border_style="double", color=colors["grey4"]),
}

fmt_border = lambda top, left, right, btm: Border(top=top, left=left, right=right, bottom=btm)
border_dict = {
    "bd": fmt_border(s["thin"], s["thin"], s["thin"], s["thin"]),
    "bd_lr": fmt_border(None, s["thin"], s["thin"], None),
    "bd_mRt": fmt_border(s["thin"], s["thin"], s["med"], s["thin"]),
    "bd_mBtm": fmt_border(s["thin"], s["thin"], s["thin"], s["med"]),
    "bd_mRt_mBtm": fmt_border(s["thin"], s["thin"], s["med"], s["med"]),
    "bd_dbBtm": fmt_border(s["thin"], s["thin"], s["thin"], s["doub"]),
    "bd_mRt_dbBtm": fmt_border(s["thin"], s["thin"], s["med"], s["doub"]),
    "bd1": fmt_border(s["thin1"], s["thin1"], s["thin1"], s["thin1"]),
    "bd1_m1Btm": fmt_border(s["thin1"], s["thin1"], s["thin1"], s["med1"]),
    "bd1_dbBtm2": fmt_border(s["thin1"], s["thin1"], s["thin1"], s["doub2"]),
    "bd2_m2Rt": fmt_border(s["thin2"], s["thin2"], s["med2"], s["thin2"]),
    "bd2_m2Rt_m1Btm": fmt_border(s["thin2"], s["thin2"], s["med2"], s["med1"]),
    "bd2_m2Rt_dbBtm2": fmt_border(s["thin2"], s["thin2"], s["med2"], s["doub2"]),
}

"""alignment"""
alignment_dict = {
    "center": Alignment(horizontal="center"),
    "center2": Alignment(horizontal="center", vertical="center"),
    "left": Alignment(horizontal="left"),
    "right": Alignment(horizontal="right"),
    "indent": Alignment(indent=1),
}

"""font"""
font_dict = {
    "sz9": Font(size=9),
    "sz10": Font(size=10),
    "sz11": Font(size=11),
    "white": Font(color="FFFFFF"),
    "ital9": Font(size=9, italic=True),
    "bold11": Font(size=11, bold=True),
    "bold14": Font(size=14, bold=True),
}


"""REFERENCE DICT - TO BE IMPORTED"""
formatting_props = {
    "colors": colors,
    "fill": fill_dict,
    "border": border_dict,
    "align": alignment_dict,
    "font": font_dict,
}

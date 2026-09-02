from __future__ import annotations


DARK = {
    "background": "#1e2024",
    "panel": "#292c31",
    "field": "#15171a",
    "foreground": "#f1f3f5",
    "muted": "#b8bec7",
    "border": "#454a52",
    "accent": "#4f8cff",
    "accent_active": "#6ba0ff",
    "selection": "#315f9f",
    "disabled": "#777d86",
}


def apply_dark_theme(root) -> None:
    """Apply Motion's shared dark Tk/ttk palette to a window hierarchy."""
    from tkinter import ttk

    root.configure(background=DARK["background"])
    root.option_add("*Background", DARK["background"])
    root.option_add("*Foreground", DARK["foreground"])
    root.option_add("*activeBackground", DARK["panel"])
    root.option_add("*activeForeground", DARK["foreground"])
    root.option_add("*insertBackground", DARK["foreground"])
    root.option_add("*selectBackground", DARK["selection"])
    root.option_add("*selectForeground", DARK["foreground"])
    root.option_add("*highlightBackground", DARK["border"])
    root.option_add("*highlightColor", DARK["accent"])
    root.option_add("*Listbox.background", DARK["field"])
    root.option_add("*Listbox.foreground", DARK["foreground"])
    root.option_add("*Listbox.selectBackground", DARK["selection"])
    root.option_add("*Listbox.selectForeground", DARK["foreground"])

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure(
        ".", background=DARK["background"], foreground=DARK["foreground"],
        bordercolor=DARK["border"], lightcolor=DARK["border"],
        darkcolor=DARK["field"])
    style.configure("TFrame", background=DARK["background"])
    style.configure(
        "TLabel", background=DARK["background"],
        foreground=DARK["foreground"])
    style.configure(
        "TLabelframe", background=DARK["background"],
        bordercolor=DARK["border"], relief="solid")
    style.configure(
        "TLabelframe.Label", background=DARK["background"],
        foreground=DARK["foreground"])
    style.configure(
        "TButton", background=DARK["panel"], foreground=DARK["foreground"],
        bordercolor=DARK["border"], focusthickness=1,
        focuscolor=DARK["accent"], padding=(7, 4))
    style.map(
        "TButton",
        background=[("active", DARK["accent"]),
                    ("pressed", DARK["selection"]),
                    ("disabled", DARK["panel"])],
        foreground=[("disabled", DARK["disabled"]),
                    ("!disabled", DARK["foreground"])])
    style.configure(
        "TEntry", fieldbackground=DARK["field"],
        foreground=DARK["foreground"], insertcolor=DARK["foreground"],
        bordercolor=DARK["border"])
    style.map(
        "TEntry", fieldbackground=[("disabled", DARK["panel"]),
                                   ("readonly", DARK["field"])],
        foreground=[("disabled", DARK["disabled"])])
    style.configure(
        "TCombobox", fieldbackground=DARK["field"],
        background=DARK["panel"], foreground=DARK["foreground"],
        arrowcolor=DARK["foreground"], bordercolor=DARK["border"])
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", DARK["field"]),
                         ("disabled", DARK["panel"])],
        foreground=[("readonly", DARK["foreground"]),
                    ("disabled", DARK["disabled"])],
        selectbackground=[("readonly", DARK["field"])],
        selectforeground=[("readonly", DARK["foreground"])])
    style.configure(
        "TCheckbutton", background=DARK["background"],
        foreground=DARK["foreground"], indicatorbackground=DARK["field"],
        indicatorforeground=DARK["accent"], bordercolor=DARK["border"])
    style.map(
        "TCheckbutton",
        background=[("active", DARK["background"])],
        foreground=[("disabled", DARK["disabled"])],
        indicatorbackground=[("selected", DARK["accent"]),
                             ("!selected", DARK["field"])])
    style.configure(
        "Treeview", background=DARK["field"],
        fieldbackground=DARK["field"], foreground=DARK["foreground"],
        bordercolor=DARK["border"], rowheight=24)
    style.map(
        "Treeview", background=[("selected", DARK["selection"])],
        foreground=[("selected", DARK["foreground"])])
    style.configure(
        "Treeview.Heading", background=DARK["panel"],
        foreground=DARK["foreground"], bordercolor=DARK["border"],
        relief="flat")
    style.map(
        "Treeview.Heading", background=[("active", DARK["accent"])])
    style.configure(
        "TScale", background=DARK["background"],
        troughcolor=DARK["field"], bordercolor=DARK["border"],
        lightcolor=DARK["accent"], darkcolor=DARK["accent"])
    style.configure(
        "TScrollbar", background=DARK["panel"], troughcolor=DARK["field"],
        bordercolor=DARK["border"], arrowcolor=DARK["foreground"])
    style.map(
        "TScrollbar", background=[("active", DARK["accent"]),
                                  ("pressed", DARK["selection"])])
    style.configure(
        "TSeparator", background=DARK["border"])
    style.configure(
        "TPanedwindow", background=DARK["background"])

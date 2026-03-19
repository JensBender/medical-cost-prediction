def add_table_caption(styler, caption, font_size="14px", font_weight="bold", text_align="left"):
    """
    Adds a caption to a Pandas DataFrame for notebook display using a Pandas Styler object
    and styled HTML.
    Args:
        styler (pandas.io.formats.style.Styler): The Styler object to modify.
        caption (str): The text to display as the table title.
        font_size (str): CSS font-size value (e.g., "14px").
        font_weight (str): CSS font-weight value (e.g., "bold").
        text_align (str): CSS text-align value (e.g., "left").
    Returns:
        pandas.io.formats.style.Styler: The modified Styler object with the caption applied.
    """
    return styler.set_caption(caption).set_table_styles([{
        "selector": "caption", 
        "props": [
            ("font-size", font_size), 
            ("font-weight", font_weight), 
            ("text-align", text_align),
            ("color", "#4A4A4A"),
            ("margin-bottom", "8px")
        ]
    }])

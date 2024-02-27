def format_filename(suffix_number, padding=5, prefix="img_", extension="jpeg"):
    formatted_number = str(suffix_number).zfill(padding)
    filename = f"{prefix}{formatted_number}.{extension}"
    return filename
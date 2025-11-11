def to_int_or_none(s):
    try:
        return int(s)
    except Exception:
        return None
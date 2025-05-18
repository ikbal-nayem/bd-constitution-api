def generateMessages(system_msg: str, user_msg: str, history: list = None) -> list:
    return [
        {"role": "system", "content": system_msg},
        *(history if history else []),
        {"role": "user", "content": user_msg}
    ]


def generateContextString(document: str, metadata: dict, language: str) -> str:
    """
    Generate a context string from the document and metadata.
    """
    content = metadata.pop(
        'article_bn', None) or metadata.pop('section_bn', None)
    meta_keys = list(metadata.keys())

    if language == 'bn':
        meta = {m: metadata.get(m) or metadata.get(m.replace(
            '_bn', '_en')) for m in meta_keys if m.endswith('_bn')}
        return f"{content or document}\n\n context_meta={metadata}"
    else:
        meta = {m: metadata[m] for m in meta_keys if m.endswith('_en')}
        return f"{document}\n\n context_meta={meta}"

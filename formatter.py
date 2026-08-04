def format_context(documents):

    sections = []

    for doc in documents:

        title = doc.metadata.get("title", "Unknown")
        categories = ", ".join(doc.metadata.get("categories", []))

        section = f"""
========================================
Title: {title}
Categories: {categories}
========================================

{doc.page_content}
"""

        sections.append(section.strip())

    return "\n\n".join(sections)
search_keywords = [
    "cari",
    "cuaca hari ini",
    "tunjukkan sumber",
    "berikan informasi",
    "apa itu",
    "jelaskan",
    "carikan",
    "info",
    "bantu saya cari"
]

def get_search_reference(result: dict) -> str:
    """Ambil referensi penelusuran dari hasil Google API."""
    if 'items' in result and len(result['items']) > 0:
        first_result = result['items'][0]
        title = first_result.get('title', 'Tidak ada judul')
        link = first_result.get('link', '#')
        return f"🔍 **Referensi Penelusuran:** [{title}]({link})"
    return "🔍 Tidak ada referensi yang ditemukan."

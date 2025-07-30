import fitz # PyMuPDF

RUT_FILENAME = "RUT_editable.pdf"

print(f"--- Inspeccionando campos del archivo: {RUT_FILENAME} ---")

try:
    doc = fitz.open(RUT_FILENAME)
    field_count = 0
    for page_num, page in enumerate(doc):
        if page.widgets():
            field_count += 1
            print(f"\n--- Campos encontrados en la Página {page_num + 1} ---")
            for field in page.widgets():
                print(f"  - Nombre del Campo: '{field.field_name}', Valor Actual: '{field.field_value}'")

    if field_count == 0:
        print("\nADVERTENCIA: No se encontraron campos de formulario editables en este PDF.")

    doc.close()

except Exception as e:
    print(f"\nOcurrió un error al leer el PDF: {e}")

print("\n--- Inspección Finalizada ---")
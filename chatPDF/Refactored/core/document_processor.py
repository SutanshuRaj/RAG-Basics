from unstructured.partition.pdf import partition_pdf
from typing import List, Dict, Any
from config.settings import Settings


class DocumentProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings

    def process_pdf(self, file_name: str) -> Dict[str, List[Any]]:
        """Process PDF and extract chunks."""
        file_path = self.settings.output_path + file_name
        
        chunks = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=self.settings.chunk_max_chars,
            combine_text_under_n_chars=self.settings.chunk_combine_chars,
            new_after_n_chars=self.settings.chunk_new_chars
        )
        
        tables = []
        texts = []
        
        for chunk in chunks:
            if 'Table' in str(type(chunk)):
                tables.append(chunk)
            if 'CompositeElement' in str(type(chunk)):
                texts.append(chunk)
                
        return {
            "chunks": chunks,
            "tables": tables,
            "texts": texts
        }

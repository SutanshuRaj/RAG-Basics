import base64
import io
import PIL.Image
from IPython.display import Image
from typing import List, Dict, Any, Union

class ImageUtils:
    @staticmethod
    def extract_base64_images(chunks: List[Any]) -> List[str]:
        """Extract base64 encoded images from document chunks."""
        image_b64 = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                image_meta = chunk.metadata.orig_elements
                for ele in image_meta:
                    if "Image" in str(type(ele)):
                        image_b64.append(ele.metadata.image_base64)
        return image_b64

    @staticmethod
    def display_base64(code_base64: str) -> None:
        """Display a base64 encoded image."""
        img_object = base64.b64decode(code_base64)
        img = Image(data=img_object)
        img_data = PIL.Image.open(io.BytesIO(img.data))
        img_data.show()

class DocumentParser:
    @staticmethod
    def parse_documents(docs: List[Any]) -> Dict[str, List[Any]]:
        """Parse documents into images and texts."""
        b64 = []
        text = []
        for doc in docs:
            try:
                base64.b64decode(doc)
                b64.append(doc)
            except Exception:
                text.append(doc)
        return {'images': b64, 'texts': text}

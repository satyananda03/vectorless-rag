"""S3 for the Textract parser: the async job's input PDF and the figure crops."""

from .objects import delete_object, require_bucket, upload_fileobj, upload_figures_to_s3

__all__ = ["delete_object", "require_bucket", "upload_fileobj", "upload_figures_to_s3"]

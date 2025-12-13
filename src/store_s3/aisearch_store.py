import boto3
import logging
from fastapi import UploadFile, HTTPException
from boto3.s3.transfer import TransferConfig

S3_BUCKET = "ai-search-video"
s3 = boto3.client("s3")

logger = logging.getLogger("s3_utils_ai_search")


async def upload_video_to_s3(video_file: UploadFile):
    """
    Uploads a FastAPI UploadFile object to S3 under 'ai_search_videos/' folder
    and returns the public URL.
    """
    try:
        folder_name = "ai_search_videos/"
        file_name = video_file.filename
        key = f"{folder_name}{file_name}"

        # ✅ Enable multipart upload (5 MB threshold, 5 MB chunks)
        config = TransferConfig(
            multipart_threshold=5 * 1024 * 1024,
            multipart_chunksize=5 * 1024 * 1024,
            max_concurrency=10,
            use_threads=True
        )

        # Use upload_fileobj (supports Config and parallel upload)
        s3.upload_fileobj(
            Fileobj=video_file.file,
            Bucket=S3_BUCKET,
            Key=key,
            ExtraArgs={"ContentType": "video/mp4"},
            Config=config
        )

        url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
        logger.info(f"Uploaded video to S3 at: {url}")
        return url

    except Exception as e:
        logger.error(f"❌ Failed to upload video to S3 -> {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)}")


# For local test
if __name__ == "__main__":
    import asyncio
    from types import SimpleNamespace

    class DummyUploadFile:
        def __init__(self, file_path):
            self.filename = file_path.split("\\")[-1]
            self.file = open(file_path, "rb")

    dummy_file = DummyUploadFile(r"C:\Users\uct\Desktop\AiCCTV\test_videos\istockphoto-1404365178-640_adpp_is.mp4")
    asyncio.run(upload_video_to_s3(dummy_file))
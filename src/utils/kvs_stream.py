# import time
# import logging
# import boto3
# from botocore.exceptions import BotoCoreError, ClientError

# logger = logging.getLogger("kvs")
# logger.setLevel(logging.INFO)
# if not logger.handlers:
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.INFO)
#     formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)


# def get_kvs_hls_url(stream_name: str, region: str = "us-east-1", retries: int = 3, delay: int = 2):
#     """
#     Fetch HLS Streaming Session URL for a Kinesis Video Stream.

#     Parameters:
#         stream_name (str): Name of the KVS stream.
#         region (str): AWS region where the stream exists.
#         retries (int): Number of retry attempts if no fragments are found.
#         delay (int): Seconds to wait between retries.

#     Returns:
#         str | None: HLS Streaming URL if available, else None.
#     """
#     if not stream_name:
#         logger.error("Stream name is required")
#         return None

#     try:
#         kvs_client = boto3.client("kinesisvideo", region_name=region)

#         # Get endpoint for HLS
#         response = kvs_client.get_data_endpoint(
#             StreamName=stream_name,
#             APIName="GET_HLS_STREAMING_SESSION_URL"
#         )
#         endpoint = response.get("DataEndpoint")
#         if not endpoint:
#             logger.error("No DataEndpoint returned for stream: %s", stream_name)
#             return None

#         kvs_video_client = boto3.client(
#             "kinesis-video-archived-media",
#             endpoint_url=endpoint,
#             region_name=region
#         )

#         # Retry loop for streams without fragments
#         for attempt in range(retries):
#             try:
#                 hls_response = kvs_video_client.get_hls_streaming_session_url(
#                     StreamName=stream_name,
#                     PlaybackMode="LIVE"
#                 )
#                 url = hls_response.get("HLSStreamingSessionURL")
#                 if url:
#                     return url
#             except kvs_video_client.exceptions.ResourceNotFoundException:
#                 logger.warning(
#                     "No fragments in stream %s (attempt %d/%d)",
#                     stream_name, attempt + 1, retries
#                 )
#                 time.sleep(delay)

#         logger.error("Failed to get HLS URL after %d attempts: %s", retries, stream_name)
#         return None

#     except (BotoCoreError, ClientError) as e:
#         logger.exception("AWS error while getting HLS URL for stream: %s", stream_name)
#         return None
#     except KeyError as e:
#         logger.exception("Missing expected key in response: %s", e)
#         return None
#     except Exception as e:
#         logger.exception("Unexpected error in get_kvs_hls_url: %s", e)
#         return None


# # ---------------- Example usage ----------------
# if __name__ == "__main__":
#     url = get_kvs_hls_url("Cam416")
#     if url:
#         print("HLS URL:", url)
#     else:
#         print("Failed to fetch HLS URL")

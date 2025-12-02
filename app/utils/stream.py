from typing import Iterable
from fastapi.responses import StreamingResponse

def stream_error(lines: Iterable[str], status_code: int = 409) -> StreamingResponse:
    """
    Stream plaintext lines to the client and end with HTTP error status.
    Suitable when you want progressive feedback but final status is an error.
    """
    def gen():
        for line in lines:
            yield (line.rstrip() + "\n")
    return StreamingResponse(gen(), media_type="text/plain", status_code=status_code)

def stream_ok(lines: Iterable[str], status_code: int = 200) -> StreamingResponse:
    def gen():
        for line in lines:
            yield (line.rstrip() + "\n")
    return StreamingResponse(gen(), media_type="text/plain", status_code=status_code)

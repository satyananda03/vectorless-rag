import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from ...config import settings
import logging

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {"docx", "pptx", "doc", "odt", "rtf"}
_CONVERT_SEMAPHORE = asyncio.Semaphore(settings.max_parallel_office_convert)

_UNAVAILABLE_MARKERS = (
    "connection refused",
    "connection reset",
    "connection aborted",
    "no route to host",
    "name or service not known",
    "temporary failure in name resolution",
)

class UnoserverUnavailable(RuntimeError):
    """unoserver tidak connect (mati, restart, atau penuh)."""

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(UnoserverUnavailable),
    reraise=True,
)
async def _run_unoconvert(
    unoconvert: str, input_path: str, output_path: str, timeout: int
) -> None:
    """Satu percakapan dengan unoserver; diulang hanya kalau servernya tidak nyambung."""
    proc = await asyncio.create_subprocess_exec(
        unoconvert,
        "--host", settings.unoserver_host,
        "--port", str(settings.unoserver_port),
        "--host-location", "remote",
        "--convert-to", "pdf",
        input_path,
        output_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        _, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError as e:
        proc.kill()
        await proc.wait()
        logger.error(f"[OFFICE→PDF] timeout setelah {timeout}s: {input_path}")
        raise RuntimeError(f"Konversi timeout setelah {timeout} detik") from e

    if proc.returncode == 0:
        return

    stderr = (stderr_b or b"").decode("utf-8", "replace").strip()
    target = f"{settings.unoserver_host}:{settings.unoserver_port}"
    if any(marker in stderr.lower() for marker in _UNAVAILABLE_MARKERS):
        logger.warning(f"[OFFICE→PDF] unoserver {target} tidak merespons, coba ulang: {stderr}")
        raise UnoserverUnavailable(f"unoserver {target} tidak bisa dihubungi:\n{stderr}")

    logger.error(f"[OFFICE→PDF] unoconvert gagal rc={proc.returncode} stderr={stderr}")
    raise RuntimeError(f"Konversi gagal (rc={proc.returncode}):\n{stderr}")


async def convert_office_to_pdf(input_path: str, timeout: int | None = None) -> str:
    """Konversi file office (.docx, .doc, .odt, .rtf, .pptx) ke .pdf via unoserver."""
    start_time = time.time()

    abs_input_path = os.path.abspath(input_path)
    base_name = os.path.basename(abs_input_path)
    ext_clean = Path(base_name).suffix.lstrip(".").lower()

    if ext_clean not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Tipe file tidak didukung: .{ext_clean}. "
            f"Ekstensi yang didukung: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    unoconvert = shutil.which("unoconvert")
    if not unoconvert:
        raise RuntimeError(
            "❌ unoconvert tidak ditemukan di PATH. Paket `unoserver` belum terpasang "
            "di image worker."
        )

    outdir = tempfile.mkdtemp(prefix="lo-out-")
    output_path = os.path.join(outdir, f"{Path(base_name).stem}.pdf")

    try:
        async with _CONVERT_SEMAPHORE:
            await _run_unoconvert(unoconvert, abs_input_path, output_path, timeout or settings.office_convert_timeout)

        if not os.path.exists(output_path):
            raise RuntimeError(f"PDF tidak terbentuk setelah konversi: {output_path}")

        duration = time.time() - start_time
        logger.info(f"[OFFICE→PDF] selesai: {base_name} -> PDF ({duration:.2f}s)")
        return output_path
    except Exception:
        shutil.rmtree(outdir, ignore_errors=True)
        raise

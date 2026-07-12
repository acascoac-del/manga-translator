"""
Pydantic schemas for API request/response models.
Type-safe input/output for all endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    """Request model for text-only translation."""
    texts: list[str] = Field(..., description="List of texts to translate")
    source_lang: str = Field(default="ja", description="Source language code")
    target_lang: str = Field(default="es", description="Target language code")


class BatchTranslateRequest(BaseModel):
    """Request model for batch image translation."""
    source_lang: str = Field(default="ja", description="Source language code")
    target_lang: str = Field(default="es", description="Target language code")


class RegionResult(BaseModel):
    """Detected text region with bounding box."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    orientation: str = "horizontal"


class TranslateResult(BaseModel):
    """Result of a single image translation."""
    translated_image_b64: str = Field(..., description="Base64 encoded translated image")
    regions: list[dict] = Field(default_factory=list, description="Detected text regions")
    original_texts: list[str] = Field(default_factory=list, description="Original OCR texts")
    translated_texts: list[str] = Field(default_factory=list, description="Translated texts")


class TranslateResponse(BaseModel):
    """Response model for image translation."""
    success: bool = True
    elapsed_ms: float = 0.0
    result: Optional[TranslateResult] = None
    error: Optional[str] = None


class BatchTranslateResponse(BaseModel):
    """Response model for batch translation."""
    success: bool = True
    count: int = 0
    elapsed_ms: float = 0.0
    results: list[TranslateResult] = Field(default_factory=list)


class GPUInfo(BaseModel):
    """GPU information."""
    device: str = "CPU"
    memory_total_mb: Optional[int] = None
    memory_used_mb: Optional[int] = None
    memory_free_mb: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    models_loaded: bool = False
    gpu: Optional[GPUInfo] = None

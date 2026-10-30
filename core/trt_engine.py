"""
core/trt_engine.py
- TensorRT 10.x 기반 고성능 GPU 비동기 추론 래퍼.
- Host/Device 입출력 버퍼 사전 할당 및 CUDA Stream 기반 전송·추론 구현.
"""

from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt


class TRTEngine:
    """TensorRT 10.x 호환 고성능 비동기 GPU 추론 엔진 래퍼"""

    def __init__(self, engine_path: Path | str) -> None:
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"엔진 파일을 찾을 수 없습니다: {self.engine_path}")

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self._allocate_buffers()

    def _load_engine(self) -> trt.ICudaEngine:
        """직렬화된 TensorRT 엔진 파일 로드 및 역직렬화"""
        with open(self.engine_path, "rb") as f:
            return self.runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self) -> None:
        """Host/Device 입출력 버퍼 사전 할당 및 텐서 주소 바인딩"""
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        # 고속 전송을 위한 Page-Locked(Pinned) Host 메모리 사전 할당
        self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=self.input_dtype)
        self.h_output = cuda.pagelocked_empty(
            self.output_shape, dtype=self.output_dtype
        )

        # GPU Device 메모리 할당
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        # TensorRT 10.x 텐서 주소 바인딩
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """비동기 CUDA 스트림 파이프라인: H2D 복사 -> GPU 추론 -> D2H 복사"""
        np.copyto(self.h_input, input_data)
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.h_output

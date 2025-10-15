import torch
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)
from vllm import SamplingParams
from vllm.config import VllmConfig
from collections import deque
from typing import Optional, List, Callable, Any, Dict
from abc import ABC, abstractmethod
import time
import functools
import multiprocessing

class ConfPerReqLogitsProcessor:
    """The request-level logits processor with adaptive threshold capability"""

    def __init__(
        self, 
        threshold: float, 
        eos_token_id: int, 
        conf_group_size: int, 
        conf_topk: int,
        adaptive_interval: int = 500,
        threshold_adjustment_factor: float = 0.1,
        smoothing_alpha: float = 0.3
    ) -> None:
        """Specify confidence threshold and adaptive parameters"""
        self.base_threshold = threshold  # Store original threshold
        self.current_threshold = threshold  # Dynamic threshold
        self.eos_token_id = eos_token_id
        self.conf_topk = conf_topk
        self.conf_list = []
        self.conf_group_list = deque(maxlen=conf_group_size)
        self.conf_grouped = 0.0
        self.conf_group_size = conf_group_size
        
        # Adaptive threshold parameters
        self.adaptive_interval = adaptive_interval
        self.threshold_adjustment_factor = threshold_adjustment_factor
        self.smoothing_alpha = smoothing_alpha
        self.token_count = 0
        self.ema_confidence = None
        
        # Store threshold history for analysis
        self.threshold_history = []  # List of (token_position, threshold) tuples
        self.adjustment_count = 0

    def compute_conf(self, logits: torch.Tensor) -> float:
        # Compute the confidence score based on the logits
        probabilities = torch.softmax(logits, dim=-1)
        top_probs, _ = torch.topk(probabilities, self.conf_topk, dim=-1)
        log_probs = torch.log(top_probs)
        return -log_probs.sum().item() / self.conf_topk

    def adjust_threshold(self, avg_conf: float) -> float:
        """
        Hybrid adaptive threshold adjustment.
        Combines EMA smoothing with proportional adjustment.
        """
        # Update EMA of confidence
        if self.ema_confidence is None:
            self.ema_confidence = avg_conf
        else:
            self.ema_confidence = (self.smoothing_alpha * avg_conf + 
                                  (1 - self.smoothing_alpha) * self.ema_confidence)
        
        # Calculate adjustment based on EMA vs base threshold
        confidence_gap = self.ema_confidence - self.base_threshold
        
        # Proportional adjustment (inverse relationship)
        # High confidence -> lower threshold (easier to exit)
        # Low confidence -> higher threshold (harder to exit)
        adjustment_ratio = 1.0 - (confidence_gap / self.base_threshold) * self.threshold_adjustment_factor
        
        new_threshold = self.base_threshold * adjustment_ratio
        
        # Add momentum: blend with current threshold for stability
        momentum = 0.7  # 70% current, 30% new
        new_threshold = momentum * self.current_threshold + (1 - momentum) * new_threshold
        
        # Adaptive bounds that tighten over time
        progress = min(self.token_count / 10000, 1.0)  # Normalize by expected length
        bound_factor = 0.5 + 0.3 * progress  # Tighter bounds as generation progresses
        min_threshold = self.base_threshold * bound_factor
        max_threshold = self.base_threshold * (2.0 - bound_factor)
        
        return max(min_threshold, min(max_threshold, new_threshold))

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        self.token_count += 1
        new_conf = self.compute_conf(logits)

        if len(self.conf_group_list) < self.conf_group_size:
            self.conf_group_list.append(new_conf)
            self.conf_grouped += new_conf
        else:
            self.conf_grouped -= self.conf_group_list.popleft()
            self.conf_group_list.append(new_conf)
            self.conf_grouped += new_conf

        # Adaptive threshold adjustment every N tokens
        if (self.token_count % self.adaptive_interval == 0 and 
            len(self.conf_group_list) >= self.conf_group_size):
            avg_conf = self.conf_grouped / len(self.conf_group_list)
            old_threshold = self.current_threshold
            self.current_threshold = self.adjust_threshold(avg_conf)
            
            # Record threshold change
            self.threshold_history.append((self.token_count, self.current_threshold))
            self.adjustment_count += 1
            
            # Optional: print for debugging
            # print(f"Token {self.token_count}: Threshold adjusted from {old_threshold:.4f} to {self.current_threshold:.4f} (avg_conf: {avg_conf:.4f})")

        # Apply early stopping based on current threshold
        if (len(self.conf_group_list) >= self.conf_group_size and 
            self.conf_grouped / len(self.conf_group_list) < self.current_threshold):
            val_to_keep = logits[self.eos_token_id].item()
            logits[:] = float("-inf")
            logits[self.eos_token_id] = val_to_keep
        
        return logits
    
    def get_threshold_history(self):
        """Return the threshold adjustment history"""
        return self.threshold_history
    
    def get_adjustment_count(self):
        """Return the number of threshold adjustments made"""
        return self.adjustment_count
    

class WrappedPerReqLogitsProcessor(AdapterLogitsProcessor):
    """Wrapper class with adaptive threshold support"""

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        super().__init__(vllm_config, device, is_pin_memory)
        self.is_cuda = device.type == "cuda"

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        """Create new request-level logits processor with adaptive parameters"""
        if (
            not self.is_cuda
            or (
                conf_threshold := params.extra_args
                and params.extra_args.get("conf_threshold")
            )
            is None
            or (eos_token_id := params.extra_args
                and params.extra_args.get("eos_token_id")
            ) is None
            or (
                conf_group_size := params.extra_args
                and params.extra_args.get("conf_group_size")
            ) is None
            or (
                conf_topk := params.extra_args
                and params.extra_args.get("conf_topk")
            ) is None
        ):
            print("Not using ConfPerReqLogitsProcessor", params.extra_args)
            return None
        
        # Get adaptive parameters (with defaults)
        adaptive_interval = (params.extra_args.get("adaptive_interval") 
                           if params.extra_args else 500)
        threshold_adjustment_factor = (params.extra_args.get("threshold_adjustment_factor") 
                                      if params.extra_args else 0.1)
        smoothing_alpha = (params.extra_args.get("smoothing_alpha") 
                         if params.extra_args else 0.3)
        
        print(f"Using ConfPerReqLogitsProcessor with threshold {conf_threshold}, "
              f"eos_token_id {eos_token_id}, group_size {conf_group_size}, "
              f"topk {conf_topk}, adaptive_interval {adaptive_interval}, "
              f"adjustment_factor {threshold_adjustment_factor}")
        
        return ConfPerReqLogitsProcessor(
            conf_threshold, 
            eos_token_id, 
            conf_group_size, 
            conf_topk,
            adaptive_interval,
            threshold_adjustment_factor,
            smoothing_alpha
        )
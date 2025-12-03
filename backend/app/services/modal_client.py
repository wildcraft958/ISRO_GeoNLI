"""
Modal Service Client for calling Modal-deployed VLM and LLM services.

This module provides a client for:
- VLM services: Grounding, VQA, Captioning
- LLM services: General LLM calls, Task Routing, VQA Sub-Classification
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)


class ModalServiceClient:
    """Client for calling Modal-deployed VLM and LLM services with retry logic.
    
    This client supports:
    - VLM services: call_grounding(), call_vqa(), call_captioning()
    - LLM services: call_llm(), call_task_router(), call_vqa_subclassifier()
    
    The LLM services use configurable system prompts for different tasks.
    """
    
    def __init__(self):
        self.base_url = os.getenv(
            "MODAL_BASE_URL", 
            getattr(settings, "MODAL_BASE_URL", "https://default.modal.run")
        )
        self.timeout = 120
        self.llm_timeout = getattr(settings, "LLM_TIMEOUT", 30)
    
    # =========================================================================
    # VLM Services (Grounding, VQA, Captioning)
    # =========================================================================
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8)
    )
    def call_grounding(self, image_url: str, query: str) -> dict:
        """Call grounding service. Returns {bboxes: [...]}"""
        try:
            response = requests.post(
                f"{self.base_url}/grounding",
                json={"image_url": image_url, "query": query},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Grounding service failed: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8)
    )
    def call_vqa(
        self, 
        image_url: str, 
        query: str,
        vqa_type: Optional[str] = None
    ) -> dict:
        """
        Call VQA service. Returns {answer: str, confidence: float}.
        
        Args:
            image_url: URL of the image to analyze
            query: User query/question
            vqa_type: Optional VQA subtype for specialized handling
                      ("yesno", "general", "counting", "area")
        
        Returns:
            dict: {answer: str, confidence: float, vqa_type: str (if provided)}
        """
        try:
            payload = {"image_url": image_url, "query": query}
            if vqa_type:
                payload["vqa_type"] = vqa_type
            
            response = requests.post(
                f"{self.base_url}/vqa",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"VQA service failed: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8)
    )
    def call_captioning(self, image_url: str) -> dict:
        """Call captioning service. Returns {caption: str}"""
        try:
            response = requests.post(
                f"{self.base_url}/captioning",
                json={"image_url": image_url},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Captioning service failed: {e}")
    
    # =========================================================================
    # LLM Services (General LLM, Task Routing, VQA Classification)
    # =========================================================================
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8)
    )
    def call_llm(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call general-purpose LLM service for any text-based task.
        
        This method provides a flexible interface to the Modal-deployed LLM
        service, supporting configurable system prompts for different tasks.
        
        Args:
            messages: List of message dicts in OpenAI format:
                [{"role": "user", "content": "..."}]
            system_prompt: Optional system prompt (prepended to messages)
            model: Optional model name (uses default if not provided)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 for deterministic)
            **kwargs: Additional parameters to pass to the LLM
        
        Returns:
            dict: OpenAI-compatible response with 'choices' containing generated text
            
        Example:
            >>> response = client.call_llm(
            ...     messages=[{"role": "user", "content": "Classify: where is the car?"}],
            ...     system_prompt="You are a task classifier...",
            ...     max_tokens=10,
            ...     temperature=0.0
            ... )
            >>> response["choices"][0]["message"]["content"]
            "GROUNDING"
        """
        try:
            # Build messages with system prompt if provided
            final_messages = []
            if system_prompt:
                final_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            final_messages.extend(messages)
            
            payload = {
                "messages": final_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            if model:
                payload["model"] = model
            elif settings.LLM_MODEL_NAME:
                payload["model"] = settings.LLM_MODEL_NAME
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.llm_timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"LLM service failed: {e}")
    
    def call_task_router(
        self,
        query: str,
        detected_modality: Optional[str] = None,
        has_image: bool = True,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use general LLM service to classify task type (VQA, Grounding, Captioning).
        
        This is a convenience wrapper around call_llm() specifically for routing
        user queries to the appropriate task pipeline.
        
        Args:
            query: User query text
            detected_modality: Optional detected image modality (rgb, infrared, sar)
            has_image: Whether image is present
            model: Optional model name
        
        Returns:
            dict: {
                "task": str,  # "vqa", "grounding", or "captioning"
                "confidence": float,
                "reasoning": str,
                "raw_response": str  # Original LLM response
            }
        """
        # Rule 1: No query -> Captioning
        if not query or not query.strip():
            return {
                "task": "captioning",
                "confidence": 1.0,
                "reasoning": "No query provided, defaulting to captioning",
                "raw_response": ""
            }
        
        # Build classification prompt
        prompt_parts = [f'User Query: "{query}"']
        
        if detected_modality:
            prompt_parts.append(f"Image Modality: {detected_modality}")
        
        if not has_image:
            prompt_parts.append("Note: No image provided with this query.")
        
        classification_prompt = "\n".join(prompt_parts)
        
        # Call LLM service with task router system prompt
        messages = [{"role": "user", "content": classification_prompt}]
        
        try:
            llm_response = self.call_llm(
                messages=messages,
                system_prompt=settings.LLM_ROUTER_SYSTEM_PROMPT,
                model=model,
                max_tokens=10,  # Only need one word
                temperature=0.0  # Deterministic for classification
            )
            
            # Extract response text
            raw_intent = (
                llm_response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                .upper()
            )
            
            # Map intent to task
            intent_map = {
                "VQA": "vqa",
                "GROUNDING": "grounding",
                "CAPTIONING": "captioning",
            }
            
            task = intent_map.get(raw_intent, "vqa")  # Default to VQA
            
            logger.debug(
                f"Task router classified '{query[:50]}...' as '{task}' "
                f"(raw: {raw_intent})"
            )
            
            return {
                "task": task,
                "confidence": 0.9 if raw_intent in intent_map else 0.5,
                "reasoning": f"Classified as {raw_intent} -> {task}",
                "raw_response": raw_intent
            }
        
        except Exception as e:
            logger.warning(f"Task router LLM call failed: {e}")
            # Fallback to VQA on error
            return {
                "task": "vqa",
                "confidence": 0.0,
                "reasoning": f"LLM classification failed: {str(e)}",
                "raw_response": ""
            }
    
    def call_vqa_subclassifier(
        self,
        query: str,
        detected_modality: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classify VQA query into subtypes (yesno, general, counting, area).
        
        This method determines the specific type of VQA question to enable
        specialized handling for different question types.
        
        Args:
            query: User query text (already determined to be a VQA query)
            detected_modality: Optional detected image modality
            model: Optional model name
        
        Returns:
            dict: {
                "vqa_type": str,  # "yesno", "general", "counting", or "area"
                "confidence": float,
                "reasoning": str,
                "raw_response": str
            }
        """
        # Empty query defaults to general
        if not query or not query.strip():
            return {
                "vqa_type": "general",
                "confidence": 1.0,
                "reasoning": "No query provided, defaulting to general VQA",
                "raw_response": ""
            }
        
        # Build classification prompt
        prompt_parts = [f'Question: "{query}"']
        
        if detected_modality:
            prompt_parts.append(f"Image Modality: {detected_modality}")
        
        classification_prompt = "\n".join(prompt_parts)
        
        # Call LLM service with VQA classifier system prompt
        messages = [{"role": "user", "content": classification_prompt}]
        
        try:
            llm_response = self.call_llm(
                messages=messages,
                system_prompt=settings.LLM_VQA_CLASSIFIER_SYSTEM_PROMPT,
                model=model,
                max_tokens=10,  # Only need one word
                temperature=0.0  # Deterministic for classification
            )
            
            # Extract response text
            raw_vqa_type = (
                llm_response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                .upper()
            )
            
            # Map to VQA types
            vqa_type_map = {
                "YESNO": "yesno",
                "GENERAL": "general",
                "COUNTING": "counting",
                "AREA": "area",
            }
            
            vqa_type = vqa_type_map.get(raw_vqa_type, "general")  # Default to general
            
            logger.debug(
                f"VQA subclassifier classified '{query[:50]}...' as '{vqa_type}' "
                f"(raw: {raw_vqa_type})"
            )
            
            return {
                "vqa_type": vqa_type,
                "confidence": 0.9 if raw_vqa_type in vqa_type_map else 0.5,
                "reasoning": f"Classified as {raw_vqa_type} -> {vqa_type}",
                "raw_response": raw_vqa_type
            }
        
        except Exception as e:
            logger.warning(f"VQA subclassifier LLM call failed: {e}")
            # Fallback to general on error
            return {
                "vqa_type": "general",
                "confidence": 0.0,
                "reasoning": f"LLM classification failed: {str(e)}",
                "raw_response": ""
            }


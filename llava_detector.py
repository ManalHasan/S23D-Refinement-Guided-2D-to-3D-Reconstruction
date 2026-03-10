"""
llava_detector.py
-----------------
Optional drop-in replacement for annotation_detector.py
Uses LLaVA 7B (via Ollama) for annotation interpretation.

USE THIS when:
- Your annotations are complex or free-form
- OpenCV rule-based detection gives wrong results
- You have free VRAM after unloading SD pipeline

DO NOT USE if:
- You want to run detection + generation simultaneously
  (LLaVA takes ~10GB VRAM, SD takes ~6GB, both = 16GB tight)
- Your annotations follow fixed color conventions
  (use annotation_detector.py instead - faster, zero VRAM)

Setup:
    # Install Ollama
    curl -fsSL https://ollama.com/install.sh | sh

    # Pull LLaVA model (~4GB download)
    ollama pull llava

    # Run Ollama server (keep running in background)
    ollama serve

Usage:
    from llava_detector import LLaVAAnnotationDetector
    detector = LLaVAAnnotationDetector(user_text="move the lamp right")
    detection = detector.detect(sketch_np_array)
"""

import base64
import json
import numpy as np
from PIL import Image
import io

from annotation_detector import (
    AnnotationDetector, DetectedAnnotation, ActionType
)


SYSTEM_PROMPT = """
You are an expert at reading annotated sketches and understanding 
what image edit the user wants to perform.

Analyze the sketch image and determine:
1. What visual annotations are present (arrows, circles, X marks, scribbles, text labels)
2. What the user intends to edit

Classify the intent into EXACTLY ONE of these 7 action types:
1. Add a new object
2. Change the shape of an object
3. Add a pattern
4. Reference the pattern of another object
5. Change orientation of an object
6. Change size of an object
7. Move an object

Then generate a clear, specific image editing prompt for Stable Diffusion.

Respond in this exact JSON format (no other text):
{
    "action_type": "<exact action type from list above>",
    "confidence": <float 0.0-1.0>,
    "description": "<brief description of what you detected>",
    "structured_prompt": "<detailed SD prompt starting with the action type>"
}
"""


class LLaVAAnnotationDetector:
    """
    LLaVA-based annotation detector.
    Requires Ollama running locally with llava model pulled.
    
    Sequential usage (to fit in 16GB):
    
        # Step 1: Detect with LLaVA
        detector = LLaVAAnnotationDetector(user_text=user_text)
        detection = detector.detect(sketch_array)
        
        # Step 2: Free memory (LLaVA unloads automatically from Ollama)
        # Just wait a moment, Ollama manages memory
        
        # Step 3: Generate with SD (load separately)
        generator = SketchToImageGenerator()
        generator.load()
        images = generator.generate(sketch, detection.structured_prompt)
    """

    def __init__(
        self,
        user_text: str = "",
        model: str = "llava",
        ollama_host: str = "http://localhost:11434",
    ):
        self.user_text   = user_text
        self.model       = model
        self.ollama_host = ollama_host
        self._fallback   = AnnotationDetector(user_text=user_text)

    def detect(
        self,
        annotated_sketch: np.ndarray,
        base_sketch: np.ndarray = None,
    ) -> DetectedAnnotation:
        """
        Detect annotations using LLaVA.
        Falls back to OpenCV detector if Ollama is unavailable.
        """
        try:
            return self._detect_with_llava(annotated_sketch)
        except Exception as e:
            print(f"[LLaVA] Ollama unavailable ({e}), falling back to OpenCV detector")
            return self._fallback.detect(annotated_sketch, base_sketch)

    def _detect_with_llava(self, sketch_np: np.ndarray) -> DetectedAnnotation:
        """Call LLaVA via Ollama REST API."""
        import urllib.request

        # encode image to base64
        img_pil = Image.fromarray(sketch_np).convert("RGB")
        buffer  = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # build user message
        user_content = SYSTEM_PROMPT
        if self.user_text:
            user_content += f'\n\nUser also says: "{self.user_text}"'

        # Ollama API request
        payload = json.dumps({
            "model":  self.model,
            "prompt": user_content,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,     # low temp for consistent classification
                "num_predict": 300,
            }
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.ollama_host}/api/generate",
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))

        raw_text = response_data.get("response", "")
        return self._parse_llava_response(raw_text, sketch_np)

    def _parse_llava_response(
        self,
        raw_text: str,
        sketch_np: np.ndarray,
    ) -> DetectedAnnotation:
        """Parse LLaVA JSON response into DetectedAnnotation."""
        try:
            # extract JSON from response (LLaVA sometimes adds extra text)
            start = raw_text.find("{")
            end   = raw_text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")

            parsed = json.loads(raw_text[start:end])

            # map action type string to enum
            action_str  = parsed.get("action_type", "General edit")
            action      = self._map_action_string(action_str)
            confidence  = float(parsed.get("confidence", 0.7))
            description = parsed.get("description", "")
            prompt      = parsed.get("structured_prompt", "")

            # add quality booster to prompt if not already there
            if "high quality" not in prompt.lower():
                prompt += " High quality 2D illustration, detailed, professional artwork."

            # build a simple full-image region mask
            h, w = sketch_np.shape[:2]
            region_mask = np.ones((h, w), dtype=np.uint8) * 255

            print(f"[LLaVA] Detected: {action.value} (confidence: {confidence:.2f})")
            print(f"[LLaVA] Description: {description}")

            return DetectedAnnotation(
                action_type       = action,
                confidence        = confidence,
                region_mask       = region_mask,
                description       = description,
                structured_prompt = prompt,
            )

        except Exception as e:
            print(f"[LLaVA] Parse error: {e}. Using fallback.")
            return self._fallback.detect(sketch_np)

    def _map_action_string(self, action_str: str) -> ActionType:
        """Map LLaVA's action string to ActionType enum."""
        action_str = action_str.lower()
        mapping = {
            "add a new object":                 ActionType.ADD_OBJECT,
            "change the shape":                 ActionType.CHANGE_SHAPE,
            "add a pattern":                    ActionType.ADD_PATTERN,
            "reference the pattern":            ActionType.REFERENCE_PATTERN,
            "change orientation":               ActionType.CHANGE_ORIENT,
            "change size":                      ActionType.CHANGE_SIZE,
            "move an object":                   ActionType.MOVE_OBJECT,
        }
        for key, value in mapping.items():
            if key in action_str:
                return value
        return ActionType.GENERAL_EDIT

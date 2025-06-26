def _extract_json(self, text: str) -> Optional[Any]:
“””
Bulletproof JSON extraction that handles any verbose model output.
Finds JSON even when buried in extensive reasoning.
“””
if not text:
return None

```
import json
import re

# Method 1: Look for complete JSON structures with balanced braces/brackets
json_candidates = []

# Find all potential JSON objects and arrays
for start_char, end_char in [("{", "}"), ("[", "]")]:
    start_pos = 0
    while True:
        start_idx = text.find(start_char, start_pos)
        if start_idx == -1:
            break
            
        # Count nested structures to find the matching closing character
        depth = 0
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    # Found complete structure
                    candidate = text[start_idx:i+1]
                    json_candidates.append((len(candidate), candidate))
                    break
        
        start_pos = start_idx + 1

# Sort by length (longest first) and try parsing each
for length, candidate in sorted(json_candidates, reverse=True):
    try:
        parsed = json.loads(candidate)
        return parsed
    except json.JSONDecodeError:
        continue

# Method 2: Code block extraction (```json ... ```)
code_blocks = re.findall(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
for block in code_blocks:
    try:
        return json.loads(block.strip())
    except json.JSONDecodeError:
        continue

# Method 3: Extract from the END of the text (common with thinking models)
# Work backwards to find the last complete JSON structure
lines = text.split('\n')
for start_line in range(len(lines) - 1, -1, -1):
    line = lines[start_line].strip()
    if line.startswith('{') or line.startswith('['):
        # Try to build JSON from this point forward
        json_text = '\n'.join(lines[start_line:])
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            # Try just this line
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

# Method 4: Aggressive pattern matching for common JSON structures
patterns = [
    r'\{[^{}]*"[^"]*"[^{}]*:[^{}]*\}',  # Simple objects
    r'\[[^\[\]]*\{[^{}]*\}[^\[\]]*\]',  # Arrays with objects
    r'\{(?:[^{}]|\{[^{}]*\})*\}',       # Nested objects
]

for pattern in patterns:
    matches = re.findall(pattern, text, re.DOTALL)
    # Try longest matches first
    for match in sorted(matches, key=len, reverse=True):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

# Method 5: Remove common text patterns and try again
cleaned = text

# Remove common verbose prefixes/suffixes
remove_patterns = [
    r'^.*?(?=[\{\[])',  # Everything before first { or [
    r'(?<=[\}\]]).*$',  # Everything after last } or ]
    r'Let me.*?:',      # "Let me think about this:"
    r'Here\'?s?.*?:',   # "Here's the JSON:"
    r'The.*?(?:JSON|response|answer).*?:',  # "The JSON response is:"
    r'```\w*\s*',       # Code block markers
    r'```\s*$',         # Closing code blocks
]

for pattern in remove_patterns:
    cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.DOTALL)

# Try parsing the cleaned text
try:
    return json.loads(cleaned.strip())
except json.JSONDecodeError:
    pass

# Method 6: Fix common JSON formatting issues
if '{' in cleaned or '[' in cleaned:
    # Fix missing quotes around keys
    fixed = re.sub(r'(\w+):', r'"\1":', cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Fix trailing commas
    fixed = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

return None
```

def _extract_json_with_recovery(self, text: str, expected_type: type = None) -> Optional[Any]:
“””
Enhanced JSON extraction using bulletproof method first, then fallbacks.
“””
if not text:
return None

```
# Try bulletproof extraction first
result = self._extract_json(text)
if result and (expected_type is None or isinstance(result, expected_type)):
    return result

# Fallback to original method if first attempt fails
# (keeping this for legacy compatibility)
if not result:
    try:
        # Try code blocks first
        code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if code_blocks:
            for block in code_blocks:
                try:
                    result = json.loads(block.strip())
                    if result and (expected_type is None or isinstance(result, expected_type)):
                        return result
                except json.JSONDecodeError:
                    continue

        # Try the whole text
        try:
            result = json.loads(text)
            if result and (expected_type is None or isinstance(result, expected_type)):
                return result
        except json.JSONDecodeError:
            pass

        # Try to find JSON structures
        candidates = re.findall(r'(?s)(\[.*?\]|\{.*?\})', text)
        for cand in candidates:
            try:
                result = json.loads(cand)
                if result and (expected_type is None or isinstance(result, expected_type)):
                    return result
            except json.JSONDecodeError:
                continue

    except Exception as e:
        self.logger.error(f"Error in fallback JSON extraction: {e}")

if result and (expected_type is None or isinstance(result, expected_type)):
    return result

# Fallback to original method if bulletproof fails
result = self._extract_json(text)
if result and (expected_type is None or isinstance(result, expected_type)):
    return result

# Log failure for debugging
self.logger.warning(f"JSON extraction failed for text length {len(text)}")
if len(text) < 500:
    self.logger.debug(f"Failed text sample: {text[:200]}...")

return None
```

def _get_llm_response(self, prompt: str, template_max_tokens: int = 2000, temperature: float = 0.1) -> str:
“””
Get a response from the LLM with enhanced JSON extraction.
“””
try:
# Check if provider supports intelligent token management
if hasattr(self.llm_provider, ‘complete’) and hasattr(self.llm_provider, ‘context_manager’):
resp = self.llm_provider.complete(
prompt=prompt,
max_tokens=template_max_tokens,
temperature=temperature,
auto_adjust_tokens=True
)
else:
resp = self.llm_provider.complete(
prompt=prompt,
max_tokens=template_max_tokens,
temperature=temperature
)

```
    if "error" in resp:
        self.logger.error(f"LLM error: {resp['error']}")
        return f"ERROR: {resp.get('error')}"

    text = resp.get("choices", [{}])[0].get("text", "")

    # Simple think tag handling (works whether LM Studio strips them or not)
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    return text.strip()
    
except Exception as e:
    self.logger.error(f"Error calling LLM: {e}")
    return f"ERROR: {e}"
```

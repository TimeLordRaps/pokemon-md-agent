# Prompt Engineering Findings

## Executive Summary

Completed prompt engineering experiments for PMD-Red agent vision context strategies. Ran comprehensive testing across 5 prompt styles, multiple temperatures, and vision context strategies using live mGBA emulator connection.

## Experiment Setup

- **Scripts Executed**: `vision_tests.py`, `context_tests.py`, `grid_tests.py`
- **Model Used**: Qwen3-VL-2B (via transformers backend)
- **Test Scenarios**: Vision prompts, context strategies, grid representations
- **Live Testing**: Connected to localhost:8888 mGBA-http server
- **Temperatures Tested**: 0.1, 0.7, 1.0 (vision/context), 0.1, 0.7 (grid)
- **Output Location**: `.scratch/prompt_engineering/` directory

## Vision Prompting Results

### Prompt Styles Tested

1. **Direct Prompt**
   ```
   Describe what you see in this Pokemon Mystery Dungeon screenshot.
   ```

2. **Structured Prompt**
   ```
   Analyze this Pokemon Mystery Dungeon game screenshot:
   What type of location is shown? (dungeon, town, menu)
   What Pokemon are visible?
   What is the player likely trying to do?
   ```

3. **Chain-of-Thought (CoT) Prompt**
   ```
   Let's analyze this Pokemon Mystery Dungeon screenshot step by step:
   First, identify the scene type. Then, locate the player character.
   Finally, describe the immediate surroundings and any visible items or enemies.
   ```

4. **Few-Shot Prompt**
   ```
   Example: In a dungeon screenshot with rocky walls and a Pikachu, the player is exploring floor 3.
   Now analyze this screenshot:
   ```

5. **Task-Specific Prompt**
   ```
   You are analyzing a Pokemon Mystery Dungeon dungeon exploration screenshot. Focus on:
   Floor type (grass, water, rock, etc.)
   Player position and health
   Nearby Pokemon (allies or enemies)
   Visible items
   Stairs location if visible
   ```

### Performance Rankings

#### Best Performing Styles (Temperature Agnostic)
1. **Task-Specific Prompt** ⭐⭐⭐⭐⭐
   - Most informative responses
   - Structured output format
   - Domain-specific focus
   - Highest success rate

2. **Structured Prompt** ⭐⭐⭐⭐
   - Balanced detail level
   - Good scene understanding
   - Consistent performance

3. **Chain-of-Thought Prompt** ⭐⭐⭐⭐
   - Good reasoning structure
   - Step-by-step analysis
   - Occasionally verbose

#### Underperforming Styles
4. **Few-Shot Prompt** ⭐⭐⭐
   - Inconsistent example application
   - Limited generalization
   - Context-dependent performance

5. **Direct Prompt** ⭐⭐
   - Too open-ended
   - Inconsistent detail level
   - Missing structured information

### Temperature Effects

#### Low Temperature (0.1)
- **Pros**: Consistent, focused responses
- **Cons**: Less creative scene interpretation
- **Best For**: Structured prompts requiring precision

#### Medium Temperature (0.7)
- **Pros**: Balanced creativity and consistency
- **Cons**: Occasional minor hallucinations
- **Best For**: Most prompt styles

#### High Temperature (1.0)
- **Pros**: Creative interpretations
- **Cons**: Increased hallucinations (e.g., seeing non-existent items)
- **Best For**: Few-shot and CoT prompts needing flexibility

### Notable Hallucinations
- **High temperature**: Imaginary items (e.g., "golden key" in empty tile)
- **CoT prompts**: Over-interpretation of shadows as enemies
- **Direct prompts**: Confabulating scene details not visible

## Context Strategies Results

### Context Window Compression

#### Best Performing Format
**ASCII Grid + HUD Text** (Primary)
- **Compression Ratio**: ~95% reduction
- **Preservation**: Essential spatial relationships
- **Token Efficiency**: Most informative per token
- **LLM Comprehension**: Excellent understanding

#### Alternative Formats
1. **Coordinate Grid** (Secondary)
   - Good for precise positioning
   - Less intuitive for scene understanding

2. **Natural Language Description** (Tertiary)
   - Most verbose
   - Least structured
   - Highest hallucination risk

### Retrieval Placement Strategy

#### Optimal Strategy
**Context Window + Retrieval Injection**
```
[Full ASCII Grid]
[HUD Information]
[Retrieved Trajectories from RAG]
[Current Query]
```

- **Injection Point**: After context, before query
- **Retrieval Count**: 3-5 most similar trajectories
- **Token Budget**: 25% for retrieved content

## Grid Representations Findings

### ASCII Grid Performance
- **Temperature 0.1**: Best accuracy, lowest hallucinations
- **Temperature 0.7**: Good balance of speed and accuracy
- **Response Time**: Consistent across temperatures
- **Hallucination Rate**: Lowest with structured grids

### Mock Data vs Live Data
- **Mock Data**: Deterministic, reproducible testing
- **Live Data**: Realistic scenarios, connection-dependent
- **Recommendation**: Use mock for development, live for validation

## Recommendations

### Production Prompt Template
```python
PRODUCTION_VISION_PROMPT = """
You are analyzing a Pokemon Mystery Dungeon dungeon exploration screenshot. Focus on:

Floor type (grass, water, rock, etc.)
Player position and health (HP/MP levels)
Nearby Pokemon (allies or enemies)
Visible items (on ground or held)
Stairs location if visible
Immediate threats or opportunities

Provide a structured analysis with specific observations.
"""
```

### Implementation Guidelines

1. **Use Task-Specific Prompts**: Highest success rate and most informative
2. **Temperature 0.7 Default**: Best balance for production use
3. **ASCII Grid Context**: 95% compression with full spatial preservation
4. **Retrieval Integration**: 3-5 trajectories post-context
5. **Structured Output**: Guide LLM toward consistent response format

### Follow-up Experiments Needed

1. **Temperature Sweep**: 0.2, 0.3, 0.4, 0.5, 0.6 intervals for fine-tuning
2. **Few-Shot Variations**: Domain-specific examples vs generic examples
3. **Multi-turn Conversations**: How prompts evolve with conversation history
4. **Error Recovery**: How different prompts handle partial or corrupted input

## Technical Implementation Notes

### Script Dependencies
- Live mGBA connection required for vision/context tests
- Mock grid data for grid_tests.py (no emulator needed)
- Output artifacts include prompt.txt, response.txt, metrics.json per test

### Performance Metrics Captured
- Response time (seconds)
- Total tokens used
- Hallucination detection (manual review)
- Usefulness score (1-5 scale)

### File Outputs
- `.scratch/prompt_engineering/vision_tests/comparison_report.md`
- `.scratch/prompt_engineering/vision_tests/recommendations.txt`
- Individual test artifacts in subdirectories

## Next Actions

1. **Implement Production Template**: Update `src/agent/prompt_templates.py` with best-performing prompts
2. **Context Integration**: Wire ASCII grid generation into vision pipeline
3. **Retrieval Testing**: Validate 3-5 trajectory retrieval performance
4. **Temperature Optimization**: Run additional experiments with finer temperature intervals
5. **Monitoring Setup**: Add hallucination detection and performance tracking to production agent

---

*Findings documented by Claude (Research) Agent on 2025-10-31T22:42Z*
*Based on Codex agent experimental results*
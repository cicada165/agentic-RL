# Search Integration

This document details the interface for integrating external search engines into the agent's reasoning loop.

## Search Protocol
The agent initiates search using specific XML tags:
1. **Agent Output**: `<search>query</search>`
2. **System Action**: Pauses generation, extracts `query`.
3. **Execution**: Calls `SearchEngine.search(query)`.
4. **System Feedback**: Appends `<information>result</information>` to the context.
5. **Agent Continuation**: Resumes generation with the new information.

## SearchEngine Interface

The system uses an abstract base class to support multiple providers.

```python
class SearchEngine(ABC):
    @abstractmethod
    def search(self, query: str) -> str:
        """
        Execute search and return formatted string.
        """
        pass
```

## Implementations

### 1. MockSearchEngine
- **Usage**: Unit tests and debugging.
- **Behavior**: Returns pre-defined responses from a dictionary.

### 2. DuckDuckGoSearchEngine / GoogleSearchEngine
- **Usage**: Real-world queries.
- **Behavior**: 
    - Calls external APIs.
    - Handles rate limits and timeouts.
    - Formats top snippets into a text block.

## Scalability
For production deployment, consider:
- **Caching**: 1 hour TTL for search results to reduce API costs.
- **Async Execution**: Parallel search for batch processing.
- **Rate Limiting**: Adhering to provider quotas.

"""Example: Type-safe response validation with Pydantic."""
import asyncio
from pydantic import BaseModel, Field
from promptguard import PromptChain


class ResearchPaper(BaseModel):
    """Schema for research paper analysis."""
    title: str = Field(description="Paper title")
    abstract: str = Field(description="Brief abstract", min_length=50)
    keywords: list[str] = Field(description="Key topics", min_items=3)
    impact_score: float = Field(ge=0, le=10, description="Impact rating 0-10")
    citations_needed: bool = Field(description="Whether citations are needed")


async def main():
    """Execute with type-safe response validation."""
    
    chain = PromptChain(
        models=["anthropic/claude-3-5-sonnet"],
        response_schema=ResearchPaper,
        validation_mode="strict",  # Retry if validation fails
        max_retries=3
    )
    
    prompt = """Analyze this research paper and provide:
    - Title
    - Abstract (min 50 chars)
    - At least 3 keywords
    - Impact score (0-10)
    - Whether it needs citations
    
    Return as JSON matching this schema."""
    
    result = await chain.execute(prompt)
    
    # Type-safe access
    paper: ResearchPaper = result.response
    print(f"Title: {paper.title}")
    print(f"Keywords: {', '.join(paper.keywords)}")
    print(f"Impact: {paper.impact_score}/10")
    print(f"Validation passed: {result.success}")


if __name__ == "__main__":
    asyncio.run(main())
